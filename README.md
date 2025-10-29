# Cued Speech Processing Tools

Python package for decoding and generating cued speech videos with MediaPipe and deep learning.

## Features

- **Decoder**: Convert cued speech videos to text with subtitles using neural networks and language models
- **Generator**: Create cued speech videos from text with automatic hand gesture overlay
- **Automatic Data Management**: Downloads required models and data automatically

## Installation

### Prerequisites
- Python 3.11.*
- Pixi (for Montreal Forced Aligner)

### Setup Steps

1. **Install Pixi**
```bash
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows PowerShell
irm https://pixi.sh/install.ps1 | iex
```

2. **Create Pixi environment**
```bash
mkdir cued-speech-env && cd cued-speech-env
pixi init
pixi add "python==3.11"
pixi add montreal-forced-aligner=3.3.4
```

3. **Install package**
```bash
pixi run python -m pip install cued-speech
```

4. **Download data and setup MFA models**
```bash
pixi shell
cued-speech download-data
pixi run mfa models save acoustic download/french_mfa.zip --overwrite
pixi run mfa models save dictionary download/french_mfa.dict --overwrite
```

5. **Verify installation**
```bash
cued-speech --help
```

## Quick Start

### Decode Video (Cued Speech → Text)
```bash
# Basic usage with default parameters, we use the provided test video
cued-speech decode

# Custom video
cued-speech decode --video_path /path/to/video.mp4
```

### Generate Video (Text → Cued Speech)
```bash
# Text extracted automatically from video audio
cued-speech generate input_video.mp4

# Skip Whisper 
cued-speech generate video.mp4 --skip-whisper --text "Votre texte ici"
```

## Command Line Options

### Decoder
**Core Options:**
- `--video_path PATH` - Input video (default: `download/test_decode.mp4`)
- `--output_path PATH` - Output video (default: `output/decoder/decoded_video.mp4`)
- `--right_speaker [True|False]` - Speaker handedness (default: `True`)
- `--auto_download [True|False]` - Auto-download data (default: `True`)

**Model Paths (optional):**
- `--model_path PATH` - TFLite CTC model (default: `download/cuedspeech_model_fixed_temporal.tflite`)
- `--vocab_path PATH` - Phoneme vocabulary
- `--face_tflite PATH` - Face landmark model (default: `download/face_landmarker.task`)
- `--hand_tflite PATH` - Hand landmark model (default: `download/hand_landmarker.task`)
- `--pose_tflite PATH` - Pose landmark model (default: `download/pose_landmarker_full.task`)

### Generator
**Options:**
- `VIDEO_PATH` (required) - Input video file
- `--text TEXT` - Manual text input (optional, otherwise extracted from audio)
- `--output_path PATH` - Output video (default: `output/generator/generated_cued_speech.mp4`)
- `--language LANG` - Language (default: `french`)
- `--skip-whisper` - Skip Whisper transcription (requires `--text`)
- `--easing TYPE` - Animation easing: `linear`, `ease_in_out_cubic`, `ease_out_elastic`, `ease_in_out_back`
- `--morphing/--no-morphing` - Hand shape morphing (default: enabled)
- `--transparency/--no-transparency` - Transparency effects (default: enabled)
- `--curving/--no-curving` - Curved trajectories (default: enabled)

## Python API

### Decoder
```python
from cued_speech import decode_video

decode_video(
    video_path="input.mp4",
    right_speaker=True,
    output_path="output/decoder/"
)
```

### Generator
```python
from cued_speech import generate_cue
import whisper

# Automatic text extraction
model = whisper.load_model("medium", download_root="download")
result_path = generate_cue(
    text=None,  # Extracted from video
    video_path="video.mp4",
    output_path="output/generator/",
    config={
        "model": model,  # Optional preloaded Whisper model
        "language": "french",
        "easing_function": "ease_in_out_cubic",
        "enable_morphing": True,
        "enable_transparency": True,
        "enable_curving": True,
    }
)

# With manual text
result_path = generate_cue(
    text="Bonjour tout le monde",
    video_path="video.mp4",
    output_path="output/generator/",
    config={"skip_whisper": True}
)
```

## Data Management

```bash
# Download all required data
cued-speech download-data

# List available data
cued-speech list-data

# Clean up data
cued-speech cleanup-data --confirm
```

### Downloaded Files

Data is stored in `./download/`:

**Decoder:**
- `cuedspeech_model_fixed_temporal.tflite` - TFLite CTC model (100-frame fixed temporal window)
- `phonelist.csv`, `lexicon.txt` - Vocabularies
- `kenlm_fr.bin`, `kenlm_ipa.binary` - Language models
- `homophones_dico.jsonl` - Homophone dictionary
- `face_landmarker.task` - Face landmarks (478 points, 3.6 MB, float16)
- `hand_landmarker.task` - Hand landmarks (21 points/hand, 7.5 MB, float16)
- `pose_landmarker_full.task` - Pose landmarks (33 points, 9.0 MB, float16, FULL complexity)

**Generator:**
- `rotated_images/` - Hand shape images
- `french_mfa.dict`, `french_mfa.zip` - MFA models

**Test Files:**
- `test_decode.mp4`, `test_generate.mp4`

## Architecture

### Decoder
- **MediaPipe Tasks API**: Latest float16 models for landmark detection (.task files)
- **TFLite CTC Model**: Three-stream fusion encoder (hand shape, position, lips) with 100-frame fixed temporal window
- **CTC Decoder**: Phoneme recognition with KenLM beam search
- **Language Model**: KenLM for French sentence correction
- **Real-time Processing**: Overlap-save windowing for streaming inference

### Generator
- **Whisper**: Speech-to-text transcription
- **MFA**: Montreal Forced Alignment for phoneme timing
- **Dynamic Scaling**: Hand size automatically adapts to face width
- **Hand Rendering**: MediaPipe-based hand landmark detection for accurate positioning

## Notes

- Models designed for 30 FPS videos
- Hand size automatically scales based on detected face width
- Decoder uses MediaPipe Tasks API (`.task` files) for landmark detection
- CTC model uses TFLite with 100-frame fixed temporal window for optimal performance

## License

MIT License - see LICENSE file

## Support

Contact: boubasow.pro@gmail.com
