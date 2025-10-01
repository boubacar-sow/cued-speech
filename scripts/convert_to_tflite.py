import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import csv


def _resolve_paths(project_root: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    vocab_path = Path(args.vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = project_root / vocab_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    return model_path, vocab_path, out_dir


class ThreeStreamFusionEncoder(torch.nn.Module):
    def __init__(self, hand_shape_dim: int, hand_pos_dim: int, lips_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.hand_shape_gru = torch.nn.GRU(hand_shape_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.hand_pos_gru = torch.nn.GRU(hand_pos_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.lips_gru = torch.nn.GRU(lips_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.fusion_gru = torch.nn.GRU(hidden_dim * 6, hidden_dim * 3, n_layers, bidirectional=True, batch_first=True)

    def forward(self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor) -> torch.Tensor:
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)
        hand_pos_out, _ = self.hand_pos_gru(hand_pos)
        lips_out, _ = self.lips_gru(lips)
        combined = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        fusion_out, _ = self.fusion_gru(combined)
        return fusion_out


class CTCModel(torch.nn.Module):
    def __init__(self, hand_shape_dim: int, hand_pos_dim: int, lips_dim: int, output_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        self.encoder = ThreeStreamFusionEncoder(hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim, n_layers)
        encoder_output_dim = hidden_dim * 6
        self.ctc_fc = torch.nn.Linear(encoder_output_dim, output_dim)

    def forward(self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(hand_shape, hand_pos, lips)
        return self.ctc_fc(enc)


class ExportWrapper(torch.nn.Module):
    """Wraps CTCModel to pass explicit zero-initial states to all GRUs.

    This avoids ConstantOfShape/Slice constructs that confuse some ONNX->TF converters.
    """
    def __init__(self, model: CTCModel):
        super().__init__()
        self.model = model

    def forward(self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor) -> torch.Tensor:
        batch_size = hand_shape.size(0)
        device = hand_shape.device

        # Stream GRUs
        hs_gru = self.model.encoder.hand_shape_gru
        hp_gru = self.model.encoder.hand_pos_gru
        lp_gru = self.model.encoder.lips_gru
        fu_gru = self.model.encoder.fusion_gru

        num_layers_directional = hs_gru.num_layers * (2 if hs_gru.bidirectional else 1)
        hidden_size = hs_gru.hidden_size
        h0_hs = torch.zeros(num_layers_directional, batch_size, hidden_size, device=device, dtype=hand_shape.dtype)
        h0_hp = torch.zeros(num_layers_directional, batch_size, hidden_size, device=device, dtype=hand_pos.dtype)
        h0_lp = torch.zeros(num_layers_directional, batch_size, hidden_size, device=device, dtype=lips.dtype)

        hand_shape_out, _ = hs_gru(hand_shape, h0_hs)
        hand_pos_out, _ = hp_gru(hand_pos, h0_hp)
        lips_out, _ = lp_gru(lips, h0_lp)

        combined = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)

        num_layers_directional_fu = fu_gru.num_layers * (2 if fu_gru.bidirectional else 1)
        hidden_size_fu = fu_gru.hidden_size
        h0_fu = torch.zeros(num_layers_directional_fu, batch_size, hidden_size_fu, device=device, dtype=combined.dtype)
        fusion_out, _ = fu_gru(combined, h0_fu)

        logits = self.model.ctc_fc(fusion_out)
        return logits


def _load_vocabulary_size(vocab_path: Path) -> int:
    with open(vocab_path, "r") as f:
        reader = csv.reader(f)
        vocabulary_list = [row[0] for row in reader]
    seen = set()
    unique_vocab = [x for x in vocabulary_list if not (x in seen or seen.add(x))]
    special_tokens = ["<BLANK>", "<UNK>", "<SOS>", "<EOS>", "<PAD>"]
    for token in reversed(special_tokens):
        if token not in unique_vocab:
            unique_vocab.insert(0, token)
    if unique_vocab[0] != "<BLANK>":
        if "<BLANK>" in unique_vocab:
            unique_vocab.remove("<BLANK>")
        unique_vocab.insert(0, "<BLANK>")
    return len(unique_vocab)


def _load_model_only(model_path: Path, vocab_path: Path) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = _load_vocabulary_size(vocab_path)
    model = CTCModel(hand_shape_dim=7, hand_pos_dim=18, lips_dim=8, output_dim=output_dim)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def export_to_onnx(model, out_dir: Path, seq_len: int, opset: int, use_dynamic_axes: bool) -> Path:
    model.eval()
    device = next(model.parameters()).device

    # Input dims as used by the model in decoder.py
    dummy_hs = torch.randn(1, seq_len, 7, device=device, dtype=torch.float32)
    dummy_hp = torch.randn(1, seq_len, 18, device=device, dtype=torch.float32)
    dummy_lp = torch.randn(1, seq_len, 8, device=device, dtype=torch.float32)

    onnx_path = out_dir / "cuedspeech_model.onnx"

    dynamic_axes = None
    if use_dynamic_axes:
        dynamic_axes = {
            "hand_shape": {0: "batch", 1: "time"},
            "hand_pos": {0: "batch", 1: "time"},
            "lips": {0: "batch", 1: "time"},
            "logits": {0: "batch", 1: "time"},
        }

    torch.onnx.export(
        model,
        (dummy_hs, dummy_hp, dummy_lp),
        str(onnx_path),
        input_names=["hand_shape", "hand_pos", "lips"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )

    return onnx_path


def onnx_to_savedmodel(onnx_path: Path, out_dir: Path) -> Path:
    # Uses onnx2tf CLI to convert ONNX -> TensorFlow SavedModel
    # SavedModel will be at: out_dir / "saved_model"
    saved_model_dir = out_dir / "saved_model"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    import subprocess

    # onnx2tf will create the SavedModel inside out_dir; we direct it explicitly
    cmd = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(saved_model_dir),
    ]

    subprocess.run(cmd, check=True)
    return saved_model_dir


def savedmodel_to_tflite(saved_model_dir: Path, out_dir: Path, float16: bool) -> Path:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    tflite_path = out_dir / ("cuedspeech_model_fp16.tflite" if float16 else "cuedspeech_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path


def export_with_ai_edge_torch(model, out_dir: Path, seq_len: int, dynamic: bool, dynamic_max_time: int) -> Path:
    """Export using AI Edge Torch directly to TFLite.

    If dynamic=True, sets time dimension T to be dynamic with bounds [1, dynamic_max_time].
    Otherwise, exports a fixed-shape model using seq_len.
    """
    import ai_edge_torch

    # Example inputs for tracing
    ex_hs = torch.randn(1, seq_len, 7, dtype=torch.float32)
    ex_hp = torch.randn(1, seq_len, 18, dtype=torch.float32)
    ex_lp = torch.randn(1, seq_len, 8, dtype=torch.float32)

    if dynamic:
        constraints = [
            torch.export.dynamic_dim(ex_hs, 1, min=1, max=dynamic_max_time),
            torch.export.dynamic_dim(ex_hp, 1, min=1, max=dynamic_max_time),
            torch.export.dynamic_dim(ex_lp, 1, min=1, max=dynamic_max_time),
        ]
        exported_program = torch.export.export(model, (ex_hs, ex_hp, ex_lp), constraints=constraints)
        edge_model = ai_edge_torch.convert(exported_program)
        tflite_path = out_dir / f"cuedspeech_model_dynamic_Tmax{dynamic_max_time}.tflite"
    else:
        edge_model = ai_edge_torch.convert(model.eval(), (ex_hs, ex_hp, ex_lp))
        tflite_path = out_dir / f"cuedspeech_model_T{seq_len}.tflite"

    edge_model.export(str(tflite_path))
    return tflite_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cued-speech PyTorch model to TFLite")
    parser.add_argument("--model_path", type=str, default="download/cuedspeech-model.pt", help="Path to cuedspeech-model.pt")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary CSV (as used by decoder)")
    parser.add_argument("--out_dir", type=str, default="output/tflite", help="Output directory for artifacts")
    parser.add_argument("--seq_len", type=int, default=50, help="Dummy sequence length for export")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--static", action="store_true", help="Export with fixed batch=1 and fixed time=seq_len (no dynamic axes)")
    parser.add_argument("--ai_edge_torch", action="store_true", help="Use AI Edge Torch to export directly to TFLite (bypasses ONNX/TF)")
    parser.add_argument("--dynamic", action="store_true", help="When using --ai_edge_torch, export with dynamic time dimension using torch.export constraints")
    parser.add_argument("--dynamic_max_time", type=int, default=1200, help="Max time steps allowed when using --dynamic (min is 1)")
    parser.add_argument("--float16", action="store_true", help="Emit Float16 TFLite model")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    # Ensure we can import from src/cued_speech
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    model_path, vocab_path, out_dir = _resolve_paths(project_root, args)

    # Lazy import to avoid hard dependency if user only wants help
    # Load model without importing heavy runtime dependencies (mediapipe, kenlm)
    base_model = _load_model_only(model_path, vocab_path)
    # Wrap with explicit initial states to simplify downstream conversion
    model = ExportWrapper(base_model)

    if args.ai_edge_torch:
        tflite_path = export_with_ai_edge_torch(model, out_dir, args.seq_len, args.dynamic, args.dynamic_max_time)
        print(f"TFLite model written: {tflite_path}")
    else:
        # Legacy pipeline: ONNX -> onnx2tf -> TFLite
        onnx_path = export_to_onnx(model, out_dir, args.seq_len, args.opset, use_dynamic_axes=not args.static)
        print(f"Exported ONNX: {onnx_path}")

        saved_model_dir = onnx_to_savedmodel(onnx_path, out_dir)
        print(f"SavedModel directory: {saved_model_dir}")

        tflite_path = savedmodel_to_tflite(saved_model_dir, out_dir, args.float16)
        print(f"TFLite model written: {tflite_path}")


if __name__ == "__main__":
    main()


