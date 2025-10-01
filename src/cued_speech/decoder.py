"""Cued Speech Decoder Module.

This module provides functionality for decoding cued speech videos and generating
subtitled output with French sentences at the bottom.
"""

import csv
import json
import math
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from itertools import groupby

import absl.logging
import cv2
import kenlm
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder

# Turn off Abseil's preinit-warning
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity(absl.logging.ERROR)

# Constants
NUM_WORKERS = 4
QUEUE_MAXSIZE = 10
INFERENCE_BUFFER_SIZE = 5
INFERENCE_INTERVAL = 15

# IPA to LIAPHON mapping for French phoneme correction
IPA_TO_LIAPHON = {
    "a": "a",
    "ə": "x",
    "ɛ": "e^",
    "œ": "x^",
    "i": "i",
    "y": "y",
    "e": "e",
    "y": "y",
    "u": "u",
    "ɔ": "o",
    "o": "o^",
    "ɑ̃": "a~",
    "ɛ̃": "e~",
    "ɔ̃": "o~",
    "œ̃": "x~",
    " ": "_",
    "b": "b",
    "c": "k",
    "d": "d",
    "f": "f",
    "ɡ": "g",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "s": "s",
    "t": "t",
    "v": "v",
    "w": "w",
    "z": "z",
    "ɥ": "h",
    "ʁ": "r",
    "ʃ": "s^",
    "ʒ": "z^",
    "ɲ": "gn",
    "ŋ": "ng",
}
LIAPHON_TO_IPA = {v: k for k, v in IPA_TO_LIAPHON.items()}

# Landmark indices
LIP_INDICES = [17, 314, 405, 321, 375, 291, 84, 181, 91, 146,
               0, 267, 269, 270, 409, 40, 37, 39, 40, 185,
               61, 78, 95, 88, 87, 14, 317, 402, 324, 308,
               80, 81, 82, 13, 312, 311, 319, 308]
HAND_INDICES = list(range(21))
FACE_INDICES = [234, 200, 214, 280, 454]


def load_vocabulary(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load phoneme-to-index mapping with special tokens."""
    with open(vocab_path, "r") as file:
        reader = csv.reader(file)
        vocabulary_list = [row[0] for row in reader]

    # Remove duplicates while preserving order
    seen = set()
    unique_vocab = [x for x in vocabulary_list if not (x in seen or seen.add(x))]

    # Add special tokens if not present - BLANK should be at index 0 for CTC
    special_tokens = ["<BLANK>", "<UNK>", "<SOS>", "<EOS>", "<PAD>"]
    for token in reversed(special_tokens):
        if token not in unique_vocab:
            unique_vocab.insert(0, token)

    # Ensure BLANK is at index 0
    if unique_vocab[0] != "<BLANK>":
        unique_vocab.remove("<BLANK>")
        unique_vocab.insert(0, "<BLANK>")

    phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(unique_vocab)}
    index_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_index.items()}

    return phoneme_to_index, index_to_phoneme


class ThreeStreamFusionEncoder(nn.Module):
    """Three-stream fusion encoder for hand shape, hand position, and lips features."""

    def __init__(
        self,
        hand_shape_dim: int,
        hand_pos_dim: int,
        lips_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.hand_shape_gru = nn.GRU(
            hand_shape_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True
        )
        self.hand_pos_gru = nn.GRU(
            hand_pos_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True
        )
        self.lips_gru = nn.GRU(
            lips_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True
        )
        self.fusion_gru = nn.GRU(
            hidden_dim * 6,
            hidden_dim * 3,
            n_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(
        self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor
    ) -> torch.Tensor:
        hand_shape_out, _ = self.hand_shape_gru(hand_shape)
        hand_pos_out, _ = self.hand_pos_gru(hand_pos)
        lips_out, _ = self.lips_gru(lips)
        combined_features = torch.cat([hand_shape_out, hand_pos_out, lips_out], dim=-1)
        fusion_out, _ = self.fusion_gru(combined_features)
        return fusion_out


class CTCModel(nn.Module):
    """CTC-only model for cued speech recognition."""

    def __init__(
        self,
        hand_shape_dim: int,
        hand_pos_dim: int,
        lips_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.encoder = ThreeStreamFusionEncoder(
            hand_shape_dim, hand_pos_dim, lips_dim, hidden_dim, n_layers
        )
        encoder_output_dim = hidden_dim * 6
        self.ctc_fc = nn.Linear(encoder_output_dim, output_dim)

    def forward(
        self, hand_shape: torch.Tensor, hand_pos: torch.Tensor, lips: torch.Tensor
    ) -> torch.Tensor:
        encoder_out = self.encoder(hand_shape, hand_pos, lips)
        ctc_logits = self.ctc_fc(encoder_out)
        return ctc_logits


def polygon_area(xs: List[float], ys: List[float]) -> float:
    """Calculate polygon area using shoelace formula."""
    xs = np.array(xs)
    ys = np.array(ys)
    x_next = np.roll(xs, -1)
    y_next = np.roll(ys, -1)
    area = 0.5 * np.abs(np.dot(xs, y_next) - np.dot(ys, x_next))
    return area


def mean_contour_curvature(points: List[Tuple[float, float]]) -> float:
    """Calculate mean curvature of a contour."""
    pts = np.array(points)
    N = pts.shape[0]
    angles = []
    for i in range(N):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[(i + 1) % N]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cosang = np.dot(v1, v2) / (norm1 * norm2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angles.append(np.arccos(cosang))
    return np.mean(angles) if angles else 0.0


def get_index_pairs(property_type: str) -> List[Tuple[int, int]]:
    """Get index pairs for hand-face or hand-hand distances."""
    index_pairs = []
    if property_type == 'shape':
        index_pairs.extend([
            (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # Wrist to finger tips
        ])
    elif property_type == 'position':
        hand_indices = [8, 9, 12]  # Index and middle fingers
        face_indices = [234, 200, 214, 454, 280]  # Specific face landmarks
        for hand_index in hand_indices:
            for face_index in face_indices:
                index_pairs.append((hand_index, face_index))
    return index_pairs


def scalar_distance(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> float:
    """Calculate scalar distance between two 3D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def extract_features_single_row(
    row: pd.Series, prev: Optional[pd.Series] = None, prev2: Optional[pd.Series] = None
) -> Dict[str, List[float]]:
    """Extract all features from a single coordinate dict (row)."""
    features = {}
    
    # Normalization factors
    f1x, f1y, f1z = row.get("face_x454"), row.get("face_y454"), row.get("face_z454")
    f2x, f2y, f2z = row.get("face_x234"), row.get("face_y234"), row.get("face_z234")
    if None in (f1x, f1y, f1z, f2x, f2y, f2z):
        return features

    face_width = scalar_distance(f1x, f1y, f1z, f2x, f2y, f2z)
    
    # Hand span
    h0x, h0y, h0z = row.get("hand_x0"), row.get("hand_y0"), row.get("hand_z0")
    h9x, h9y, h9z = row.get("hand_x9"), row.get("hand_y9"), row.get("hand_z9")
    if None in (h0x, h0y, h9x, h9y, h0z, h9z):
        hand_span = face_width
    else:
        hand_span = scalar_distance(h0x, h0y, h0z, h9x, h9y, h9z)
        if hand_span == 0:
            hand_span = face_width

    # Hand-face distances & angles
    for h, f in get_index_pairs("position"):
        hx, hy, hz = row.get(f"hand_x{h}"), row.get(f"hand_y{h}"), row.get(f"hand_z{h}")
        fx, fy, fz = row.get(f"face_x{f}"), row.get(f"face_y{f}"), row.get(f"face_z{f}")
        if None not in (hx, hy, hz, fx, fy, fz):
            d = scalar_distance(hx, hy, hz, fx, fy, fz) / face_width
            features[f"distance_face{f}_hand{h}"] = d
            if f == 200:
                dx = (fx - hx) / face_width
                dy = (fy - hy) / face_width
                features[f"angle_face{f}_hand{h}"] = np.arctan2(dy, dx)

    # Hand-hand distances
    for h1, h2 in get_index_pairs("shape"):
        x1, y1, z1 = row.get(f"hand_x{h1}"), row.get(f"hand_y{h1}"), row.get(f"hand_z{h1}")
        x2, y2, z2 = row.get(f"hand_x{h2}"), row.get(f"hand_y{h2}"), row.get(f"hand_z{h2}")
        if None not in (x1, y1, z1, x2, y2, z2):
            d = scalar_distance(x1, y1, z1, x2, y2, z2) / hand_span
            features[f"distance_hand{h1}_hand{h2}"] = d
            
    # Thumb-index angle
    coords = [row.get(k) for k in ("hand_x4", "hand_y4", "hand_z4", "hand_x0", "hand_y0", "hand_z0", "hand_x8", "hand_y8", "hand_z8")]
    if None not in coords:
        x1, y1, z1, x2, y2, z2, x3, y3, z3 = coords
        def get_angle_single(x1, y1, z1, x2, y2, z2, x3, y3, z3):
            v1 = [x1-x2, y1-y2, z1-z2]
            v2 = [x3-x2, y3-y2, z3-z2]
            dot = sum(a*b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a*a for a in v1))
            norm2 = math.sqrt(sum(a*a for a in v2))
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = dot / (norm1 * norm2)
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
                return math.acos(cos_angle)
            return 0.0
        features["thumb_index_angle"] = get_angle_single(x1, y1, z1, x2, y2, z2, x3, y3, z3)
        
    # Lip metrics
    lx61, ly61, lz61 = row.get("lip_x61"), row.get("lip_y61"), row.get("lip_z61")
    lx291, ly291, lz291 = row.get("lip_x291"), row.get("lip_y291"), row.get("lip_z291")
    if None not in (lx61, ly61, lz61, lx291, ly291, lz291):
        features["lip_width"] = scalar_distance(lx61, ly61, lz61, lx291, ly291, lz291) / face_width
    
    lx0, ly0, lz0 = row.get("lip_x0"), row.get("lip_y0"), row.get("lip_z0")
    lx17, ly17, lz17 = row.get("lip_x17"), row.get("lip_y17"), row.get("lip_z17")
    if None not in (lx0, ly0, lz0, lx17, ly17, lz17):
        features["lip_height"] = scalar_distance(lx0, ly0, lz0, lx17, ly17, lz17) / face_width
    
    # Lip area and curvature
    outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    pts = [(row.get(f"lip_x{i}"), row.get(f"lip_y{i}")) for i in outer]
    if all(p[0] is not None and p[1] is not None for p in pts):
        area = polygon_area([p[0] for p in pts], [p[1] for p in pts])
        features["lip_area"] = area / (face_width ** 2)
        features["lip_curvature"] = mean_contour_curvature(pts)

    # Motion features
    if prev is not None:
        plx0, ply0 = prev.get("lip_x0"), prev.get("lip_y0")
        if None not in (lx0, ly0, plx0, ply0):
            features["lip_velocity_x"] = (lx0 - plx0) / face_width
            features["lip_velocity_y"] = (ly0 - ply0) / face_width
        
        hx8, hy8 = row.get("hand_x8"), row.get("hand_y8")
        phx8, phy8 = prev.get("hand_x8"), prev.get("hand_y8")
        if None not in (hx8, hy8, phx8, phy8):
            features["hand8_velocity_x"] = (hx8 - phx8) / hand_span
            features["hand8_velocity_y"] = (hy8 - phy8) / hand_span
        
        # Acceleration
        if prev2 is not None:
            pplx0, pply0 = prev2.get("lip_x0"), prev2.get("lip_y0")
            if None not in (plx0, pplx0):
                prev_vel_x = (plx0 - pplx0) / face_width
                features["lip_acceleration_x"] = features.get("lip_velocity_x", 0) - prev_vel_x
            if None not in (ply0, pply0):
                prev_vel_y = (ply0 - pply0) / face_width
                features["lip_acceleration_y"] = features.get("lip_velocity_y", 0) - prev_vel_y

    return features


def load_model(
    model_path: str, vocab_path: str
) -> Tuple[CTCModel, Dict[str, int], Dict[int, str]]:
    """Load the trained model and vocabulary."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    phoneme_to_index, index_to_phoneme = load_vocabulary(vocab_path)

    # Initialize model with correct dimensions matching ACSR
    hand_shape_dim = 7  # hand-hand distances + thumb-index angle
    hand_pos_dim = 18   # hand-face distances + angles
    lips_dim = 8        # lip metrics + area + curvature + velocities + accelerations
    output_dim = len(phoneme_to_index)

    model = CTCModel(hand_shape_dim, hand_pos_dim, lips_dim, output_dim)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, phoneme_to_index, index_to_phoneme


def beam_search(
    homophone_lists: List[List[str]], lm_model: kenlm.Model, beam_width: int = 20
) -> List[str]:
    """Perform beam search using KenLM language model."""
    # Initialize KenLM state at begin-sentence
    state_in = kenlm.State()
    lm_model.BeginSentenceWrite(state_in)

    # beams: list of tuples (cum_log10_score, state, word_seq)
    beams = [(0.0, state_in, [])]

    for homos in homophone_lists:
        new_beams = []
        for cum_score, state, seq in beams:
            for w in homos:
                out_state = kenlm.State()
                # BaseScore returns the log10-prob P(w | state)
                inc_score = lm_model.BaseScore(state, w, out_state)
                new_beams.append((cum_score + inc_score, out_state, seq + [w]))
        # keep only top beam_width hypotheses by log10 score
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    # return the best word sequence
    return beams[0][2]


def process_decoded_sequence(
    decoded_sequence: str, liaphon_dict: Dict[str, str]
) -> str:
    """Process decoded sequence to convert LIAPHON to French words."""
    words = decoded_sequence.split()
    french_words = []

    for word in words:
        if word in liaphon_dict:
            french_words.append(liaphon_dict[word])
        else:
            french_words.append(word)

    return " ".join(french_words)


def correct_french_sentences(
    liaphon_decoded: List[List[str]], homophones_path: str, lm_path: str
) -> List[str]:
    """Correct French sentences using homophones and language model."""
    # Convert LIAPHON to IPA
    ipa_decoded = []
    for decoded in liaphon_decoded:
        ipa_sequence = [LIAPHON_TO_IPA.get(phone, phone) for phone in decoded]
        ipa_decoded.append(ipa_sequence)
    
    # Join IPA sequences into strings
    ipa_decoded = ["".join(sequence) for sequence in ipa_decoded]

    # Load homophones mapping
    ipa_to_homophones = {}
    with open(homophones_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ipa = entry["ipa"]
            words = entry.get("words", [])
            # if no homophones, keep the IPA token itself as a placeholder
            ipa_to_homophones[ipa] = words if words else [ipa]

    # Load KenLM model
    lm = kenlm.Model(lm_path)

    # Apply beam search correction
    all_beam_decoded = []
    for ipa_sentence in ipa_decoded:
        tokens = ipa_sentence.split()
        homophone_lists = [
            ipa_to_homophones.get(tok, [tok])  # safe fallback to the token itself
            for tok in tokens
        ]
        best_seq = beam_search(homophone_lists, lm, beam_width=20)
        all_beam_decoded.append(best_seq)

    # Join into French sentences
    french_beam_sentences = [" ".join(seq).capitalize() for seq in all_beam_decoded]
    # add a full stop at the end of the sentence if it doesn't have one
    french_beam_sentences = [sentence + "." if not sentence.endswith(".") else sentence for sentence in french_beam_sentences]
    return french_beam_sentences


def write_subtitled_video(
    input_path: str, recognition_results: deque, output_path: str, fps: float
) -> None:
    """Write subtitled video with decoded French sentences at the bottom."""
    import unicodedata

    def remove_accents(text: str) -> str:
        """Remove accents from text for OpenCV compatibility."""
        nfd = unicodedata.normalize("NFD", text)
        without_accents = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
        return without_accents

    # Load phoneme mapping for IPA to French conversion
    liaphon_to_french = {}
    try:
        # Try to load from download folder first
        from .data_manager import get_data_file_path
        ipa_file_path = get_data_file_path("ipa_to_french")
        if ipa_file_path:
            df_liaphon = pd.read_csv(ipa_file_path)
        else:
            # Fallback to hardcoded path
            df_liaphon = pd.read_csv("download/ipa_to_french.csv")
        df_liaphon.dropna(inplace=True)
        liaphon_to_french = dict(zip(df_liaphon["liaphon"], df_liaphon["french"]))
    except FileNotFoundError:
        print("Warning: Could not load IPA to French mapping file")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) & ~1  # make even
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) & ~1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Sort results by frame
    results = sorted(list(recognition_results), key=lambda r: r["frame"])
    idx = 0
    current_text = ""
    next_frame_update = results[0]["frame"] if results else float("inf")

    # Text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Update decoded subtitle when reaching next inference frame
        if frame_num >= next_frame_update and idx < len(results):
            entry = results[idx]
            if "french_sentence" in entry:
                # Use corrected French sentence if available
                current_text = remove_accents(entry["french_sentence"])
            else:
                # Fallback to phoneme sequence
                decoded_seq = entry["phonemes"]
                if isinstance(decoded_seq, list):
                    decoded_seq = " ".join(decoded_seq)
                decoded_seq = process_decoded_sequence(decoded_seq, liaphon_to_french)
                current_text = remove_accents(decoded_seq)
            idx += 1
            next_frame_update = (
                results[idx]["frame"] if idx < len(results) else float("inf")
            )

        # Draw decoded subtitle at bottom
        text_size = cv2.getTextSize(current_text, font, font_scale, thickness)[0]
        x = (width - text_size[0]) // 2
        y = int(height * 0.9)

        # Draw text with black outline and white fill
        cv2.putText(
            frame,
            current_text,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            current_text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        writer.write(frame)

    cap.release()
    writer.release()

    # Re-attach original audio
    temp_output = output_path.replace(".mp4", "_noaudio.mp4")
    os.rename(output_path, temp_output)

    final_command = (
        f'ffmpeg -y -i "{temp_output}" -i "{input_path}" '
        f'-c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"'
    )
    result = os.system(final_command)
    if result == 0:
        os.remove(temp_output)
        print(f"Subtitled video with audio written to {output_path}")
    else:
        # If ffmpeg fails, keep the video without audio
        os.rename(temp_output, output_path)
        print(f"Subtitled video (without audio) written to {output_path}")
        print("Note: ffmpeg not available, audio could not be attached")


def decode_video(
    video_path: str,
    right_speaker: str,
    model_path: str,
    output_path: str,
    vocab_path: str,
    lexicon_path: str,
    kenlm_model_path: str,
    homophones_path: str,
    lm_path: str,
) -> None:
    """
    Decode a cued speech video and produce a subtitled video with French sentences at the bottom.

    Args:
        video_path: Path to input cued-speech video
        right_speaker: Which side is the speaker's hand overlay ("speaker1" or "speaker2")
        model_path: Path to the pretrained model file
        output_path: Path to save subtitled video
        vocab_path: Path to vocabulary file
        lexicon_path: Path to lexicon file
        kenlm_model_path: Path to KenLM model file
        homophones_path: Path to homophones JSONL file
        lm_path: Path to language model file
    """
    # Load model and vocabulary
    model, phoneme_to_index, index_to_phoneme = load_model(model_path, vocab_path)
    device = next(model.parameters()).device
    
    # Initialize CTC beam decoder
    tokens = list(phoneme_to_index.keys())
    # Add blank token if not present
    blank_idx = phoneme_to_index["<BLANK>"]
    blank_token = index_to_phoneme[blank_idx]
    if blank_token not in tokens:
        tokens.append(blank_token)
    
    beam_decoder = ctc_decoder(
        lexicon=lexicon_path,
        tokens=tokens,
        lm=lm_path,
        nbest=1,
        beam_size=40,
        lm_weight=3.23,
        word_score=0,
        blank_token='<BLANK>',
        sil_score=0,
        beam_threshold=50,
        sil_token='_',
        unk_word='<UNK>'
    )

    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recognition_results = deque()

    # Process video frames
    frame_count = 0
    feature_history = []
    coordinate_buffer = deque(maxlen=INFERENCE_BUFFER_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame with MediaPipe
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Extract landmarks
        landmarks_data = {"frame_number": frame_count}

        # Hand landmarks
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                landmarks_data[f"hand_x{i}"] = landmark.x
                landmarks_data[f"hand_y{i}"] = landmark.y
                landmarks_data[f"hand_z{i}"] = landmark.z

        # Face landmarks
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks_data[f"face_x{i}"] = landmark.x
                landmarks_data[f"face_y{i}"] = landmark.y
                landmarks_data[f"face_z{i}"] = landmark.z
                
        # Lip landmarks (same as face but with lip_ prefix for ACSR compatibility)
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                landmarks_data[f"lip_x{i}"] = landmark.x
                landmarks_data[f"lip_y{i}"] = landmark.y
                landmarks_data[f"lip_z{i}"] = landmark.z

        # Convert to DataFrame row
        row = pd.Series(landmarks_data)
        coordinate_buffer.append(row)

        # Extract features
        prev = coordinate_buffer[-2] if len(coordinate_buffer) >= 2 else None
        prev2 = coordinate_buffer[-3] if len(coordinate_buffer) >= 3 else None
        features = extract_features_single_row(row, prev, prev2)

        # Append to feature history
        feature_history.append(features)

        # Run inference every INFERENCE_INTERVAL frames
        if frame_count % INFERENCE_INTERVAL == 0 and len(feature_history) > 0:
            # Take the last M frames (like ACSR)
            chunk = list(feature_history)
            
            # Convert to DataFrame like ACSR
            df = pd.DataFrame(chunk).dropna()
            if df.empty:
                continue

            # Prepare inputs
            hs_cols = [c for c in df.columns if 'hand' in c and 'face' not in c]
            hp_cols = [c for c in df.columns if 'face' in c]
            lp_cols = [c for c in df.columns if 'lip' in c]

            if len(hs_cols) != 7 or len(hp_cols) != 18 or len(lp_cols) != 8:
                print(f"Warning: Unexpected feature dims - hs:{len(hs_cols)}, hp:{len(hp_cols)}, lp:{len(lp_cols)}")
                continue

            Xhs = torch.tensor(df[hs_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            Xhp = torch.tensor(df[hp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            Xlp = torch.tensor(df[lp_cols].values, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass over the chunk
            with torch.no_grad():
                logits = model(Xhs, Xhp, Xlp)[0]  # (T_chunk, V)
                log_probs = F.log_softmax(logits, dim=1)

            # Greedy baseline: remove blanks and repeats
            greedy_idxs = log_probs.argmax(axis=-1)
            greedy = []
            prev = None
            for idx in greedy_idxs:
                ph = index_to_phoneme[idx.item()]
                if ph == '<BLANK>' or idx == prev:
                    prev = idx
                    continue
                greedy.append(ph)
                prev = idx

            print("Number of frames: ", Xlp.size(1))
            print("Greedy decoded:", greedy)

            # Beam decode on these log_probs
            results = beam_decoder(log_probs.unsqueeze(0))
            if results and results[0]:
                best = results[0][0]
                pred_tokens = beam_decoder.idxs_to_tokens(best.tokens)[1:-1]
                if len(pred_tokens) > 0: 
                    if pred_tokens[-1] == '_': 
                        pred_tokens = pred_tokens[:-1] 
            else:
                # fallback greedy
                argmax = log_probs.argmax(dim=1).tolist()
                pred_tokens = [index_to_phoneme[i] for i, _ in groupby(argmax) if i != phoneme_to_index['<BLANK>']]

            # Output
            frame_id = len(chunk)
            print(f"Decoded up to frame {frame_id}: {pred_tokens}")

            if pred_tokens:
                recognition_results.append({
                    'frame': frame_id,
                    'phonemes': pred_tokens,
                })

    cap.release()
    holistic.close()

    # Apply French sentence correction
    if recognition_results:
        liaphon_sequences = [result["phonemes"] for result in recognition_results]
        corrected_sentences = correct_french_sentences(
            liaphon_sequences, homophones_path, kenlm_model_path
        )

        # Update results with corrected sentences
        for i, result in enumerate(recognition_results):
            if i < len(corrected_sentences):
                result["french_sentence"] = corrected_sentences[i]
                print(f"Frame {result['frame']}: French sentence: {corrected_sentences[i]}")

    # Write subtitled video
    write_subtitled_video(video_path, recognition_results, output_path, fps)

    print(f"Decoding complete. Output saved to: {output_path}")
