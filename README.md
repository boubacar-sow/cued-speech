# Cued Speech Processing Tools

A comprehensive Python package for processing cued speech videos with both decoding and generation capabilities. This package provides functionality to decode cued speech videos into subtitled output and generate cued speech videos from text input.

## Features

### Decoder Features
- **Real-time Video Processing**: Process cued speech videos using MediaPipe Tasks or MediaPipe Holistic for landmark extraction
- **TFLite Model Support**: Native support for MediaPipe `.task` files (float16, latest models)
- **Flexible Model Loading**: Automatically detects and uses either `.task` (MediaPipe Tasks API) or `.tflite` (TFLite Interpreter) files
- **Neural Network Inference**: Use trained CTC models for phoneme recognition
- **French Language Correction**: Apply KenLM language models and homophone correction
- **Subtitle Generation**: Generate subtitled videos with French sentences

### Generator Features
- **Text-to-Cued Speech**: Generate cued speech videos from French text input
- **Whisper Integration**: Automatic speech recognition for accurate alignment
- **MFA Alignment**: Montreal Forced Alignment for precise phoneme timing
- **Hand Gesture Overlay**: Realistic hand shape and position rendering
- **Automatic Synchronization**: Perfect alignment between speech and visual cues

### Data Management Features
- **Automatic Data Download**: Automatically download required model files and data
- **GitHub Release Integration**: Seamless download from GitHub releases
- **Smart Caching**: Avoid re-downloading existing files
- **Easy Cleanup**: Simple commands to manage downloaded data

### General Features
- **Command Line Interface**: Easy-to-use CLI for both decoding and generation
- **Organized Output Structure**: Separate folders for decoder and generator outputs
- **Extensible Architecture**: Modular design for future enhancements
- **PyPI Ready**: Ready for publication and easy installation

## Installation

### Prerequisites

- Python 3.11.*
- Pixi (to install Montreal Forced Aligner)

### Install with Pixi (Recommended)

Use Pixi to install MFA, then install `cued_speech` via pip inside the Pixi environment.

#### 1) Install Pixi

- macOS/Linux:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

- Windows (PowerShell):
```powershell
irm https://pixi.sh/install.ps1 | iex
```

More options: https://pixi.sh/installation/

#### 2) Create a clean Pixi environment and install MFA
```bash
mkdir cued-speech-env && cd cued-speech-env
pixi init
pixi add montreal-forced-aligner=3.3.4
pixi run mfa --version
```

#### 3) Install the cued_speech package (pip inside Pixi)

```bash
pixi run python -m pip install cued-speech
```

#### 4) Prepare French MFA Models (Required for Generation)

The cued speech generator requires French MFA models (acoustic + dictionary). These are now bundled with the data downloaded by the package. Just download the data, then save the models with MFA:

```bash
# Download all required data (includes MFA French models under ./download/)
pixi shell
cued-speech download-data

# Save the French acoustic model to MFA's model store (zip file)
pixi run mfa models save acoustic download/french_mfa.zip --overwrite

# Save the French dictionary model to MFA's model store (.dict file)
pixi run mfa models save dictionary download/french_mfa.dict --overwrite
```

Note:
- You can run the above inside a Pixi shell (`pixi shell`) or prefix with `pixi run` as shown.
- After saving, MFA will manage models in its own cache (e.g., `~/.local/share/mfa/models/`).

#### 5) Verify installation and see available options
```bash
pixi shell
cued-speech
```

## Data Setup

The package requires several model files and data for operation. These are automatically downloaded on first use, but you can also manage them manually.

### Manual Data Management

You can manage data files manually using the provided commands:

```bash
# Download all required data files, verify that you are in the pixi environment
cued-speech download-data 

# List available data files
cued-speech list-data

# Clean up downloaded data files
cued-speech cleanup-data --confirm
```

### Required Data Files

The following files are automatically downloaded to a `download/` folder in your current working directory:

**Core Decoder Files:**
- `cuedspeech-model.pt` - Pre-trained neural network model for phoneme recognition
- `phonelist.csv` - Phoneme vocabulary
- `lexicon.txt` - French lexicon
- `kenlm_fr.bin` - French language model
- `homophones_dico.jsonl` - Homophone dictionary
- `kenlm_ipa.binary` - IPA language model
- `ipa_to_french.csv` - IPA to French mapping

**MediaPipe TFLite Models (float16, latest):**
- `face_landmarker.task` - Face landmark detection model (478 landmarks, 3.6 MB)
- `hand_landmarker.task` - Hand landmark detection model (21 landmarks per hand, 7.5 MB)
- `pose_landmarker_full.task` - Pose landmark detection model, FULL complexity (33 landmarks, 9.0 MB)

**Generator Files:**
- `rotated_images/` - Directory containing hand shape images for generation
- `french_mfa.dict` - MFA dictionary
- `french_mfa.zip` - MFA acoustic model

**Test Files:**
- `test_decode.mp4` - Sample video for testing decoder
- `test_generate.mp4` - Sample video for testing generator

**Note:** All data files (including TFLite models) are stored in `./download/` relative to where you run the commands, making them easy to find and manage.

## Usage

### Command Line Interface

The package provides a comprehensive command-line interface for both decoding and generating cued speech videos:

Note: The models are designed for videos at 30 FPS. For best results, use input videos that are 30 FPS.

#### Decoding (Cued Speech → Text)

Decode a cued speech video into a subtitled video. The decoder uses MediaPipe Tasks API with the latest float16 models for optimal accuracy.

**Core Options:**
- `--video_path PATH` (default: `download/test_decode.mp4`): Input cued-speech video
- `--right_speaker [True|False]` (default: `True`): Whether the speaker uses the right hand
- `--output_path PATH` (default: `output/decoder/decoded_video.mp4`): Output subtitled video
- `--auto_download [True|False]` (default: `True`): Auto-download missing data files

**Model Paths:**
- `--model_path PATH` (default: `download/cuedspeech-model.pt`): Pretrained neural network model
- `--vocab_path PATH` (default: `download/phonelist.csv`): Vocabulary file
- `--lexicon_path PATH` (default: `download/lexicon.txt`): Lexicon file
- `--kenlm_fr PATH` (default: `download/kenlm_fr.bin`): KenLM model file
- `--homophones_path PATH` (default: `download/homophones_dico.jsonl`): Homophones dictionary
- `--kenlm_ipa PATH` (default: `download/kenlm_ipa.binary`): IPA language model

**TFLite Model Paths (MediaPipe Tasks):**
- `--face_tflite PATH` (default: `download/face_landmarker.task`): Face landmark model (`.task` or `.tflite`)
- `--hand_tflite PATH` (default: `download/hand_landmarker.task`): Hand landmark model (`.task` or `.tflite`)
- `--pose_tflite PATH` (default: `download/pose_landmarker_full.task`): Pose landmark model (`.task` or `.tflite`)

```bash
# Basic usage (uses default paths, automatically downloads data if needed)
cued-speech decode

# With custom video path
cued-speech decode --video_path /path/to/your/video.mp4

# Disable automatic data download
cued-speech decode --auto_download False

# Advanced usage with custom TFLite models
cued-speech decode \
    --video_path /path/to/your/video.mp4 \
    --face_tflite /path/to/face_model.task \
    --hand_tflite /path/to/hand_model.task \
    --pose_tflite /path/to/pose_model.task

# Full custom configuration
cued-speech decode \
    --video_path /path/to/your/video.mp4 \
    --output_path output/decoder/my_decoded_video.mp4 \
    --model_path /path/to/custom_model.pt \
    --vocab_path /path/to/custom_vocab.csv \
    --lexicon_path /path/to/custom_lexicon.txt \
    --kenlm_fr /path/to/custom_kenlm.bin \
    --homophones_path /path/to/custom_homophones.jsonl \
    --kenlm_ipa /path/to/custom_lm.binary \
    --face_tflite /path/to/face_model.task \
    --hand_tflite /path/to/hand_model.task \
    --pose_tflite /path/to/pose_model.task \
    --right_speaker True
```

**Note on TFLite Models:**
- The decoder automatically detects file extensions: `.task` files use MediaPipe Tasks API, `.tflite` files use TFLite Interpreter
- If TFLite models fail to load, the decoder automatically falls back to MediaPipe Holistic
- Models are downloaded automatically with `cued-speech download-data`

#### Generation (Video → Cued Speech)

Generate a cued speech video from a video file. Text is extracted with Whisper unless `--skip-whisper` is used and `--text` is provided.

Arguments:
- `VIDEO_PATH` (positional): Path to input video file

Options:
- `--text TEXT` (default: None): Provide text manually (otherwise Whisper extracts it)
- `--output_path PATH` (default: `output/generator/generated_cued_speech.mp4`): Output video path
- `--audio_path PATH` (default: None): Optional audio file (extracted from video if not provided)
- `--language [french|...]` (default: `french`): Processing language
- `--skip-whisper` (flag): Skip Whisper download/transcription (requires `--text`)
- `--easing [linear|ease_in_out_cubic|ease_out_elastic|ease_in_out_back]` (default: `ease_in_out_cubic`): Gesture easing
- `--morphing/--no-morphing` (default: `--morphing`): Hand shape morphing
- `--transparency/--no-transparency` (default: `--transparency`): Transparency effects during transitions
- `--curving/--no-curving` (default: `--curving`): Curved trajectories

```bash
# Basic usage (text extracted automatically from video)
cued-speech generate input_video.mp4

# With custom output path
cued-speech generate speaker_video.mp4 --output_path output/generator/my_generated_video.mp4

# With custom audio file
cued-speech generate speaker_video.mp4 --audio_path custom_audio.wav

# With different language
cued-speech generate speaker_video.mp4 --language english

# With manual text (optional)
cued-speech generate speaker_video.mp4 --text "Merci beaucoup pour votre attention"

# Skip Whisper if you have SSL issues
cued-speech generate speaker_video.mp4 --skip-whisper --text "Merci beaucoup pour votre attention"
```

### Output Structure

The package organizes outputs in a structured way:

```
output/
├── decoder/           # Decoded videos with subtitles
│   └── decoded_video.mp4
└── generator/         # Generated cued speech videos
    ├── audio.wav           # Extracted/processed audio
    ├── audio.TextGrid      # MFA alignment results
    ├── rendered_video.mp4  # Video with hand cues (no audio)
    ├── final_rendered_video.mp4  # Final output with audio
    └── mfa_input/          # MFA temporary files
```

### Python API

You can also use the package programmatically:

#### Decoder API

```python
from cued_speech import decode_video

# Decode a cued speech video
decode_video(
    video_path="input.mp4",
    right_speaker=True,
    model_path="/path/to/model.pt",
    output_path="output/decoder/decoded.mp4",
    vocab_path="/path/to/vocab.csv",
    lexicon_path="/path/to/lexicon.txt",
    kenlm_model_path="/path/to/kenlm.bin",
    homophones_path="/path/to/homophones.jsonl",
    lm_path="/path/to/lm.binary"
)
```

#### Generator API

```python
from cued_speech import generate_cue

# Generate a cued speech video (text extracted automatically)
result_path = generate_cue(
    text=None,  # Will be extracted from video using Whisper
    video_path="speaker_video.mp4",
    output_path="output/generator/generated.mp4",
    audio_path=None,  # Will extract from video
    config={
        "language": "french",
        "hand_scale_factor": 0.75,
        "video_codec": "libx264",
        "audio_codec": "aac"
    }
)
print(f"Generated video saved to: {result_path}")

# Or with manual text
result_path = generate_cue(
    text="Bonjour tout le monde",
    video_path="speaker_video.mp4",
    output_path="output/generator/generated.mp4"
)
```

## Architecture

### Core Components

#### Decoder Components
1. **MediaPipe Integration**: 
   - **MediaPipe Tasks API** (default): Uses latest float16 models with native `.task` file support
   - **MediaPipe Holistic** (fallback): Traditional MediaPipe solution
   - Automatic model detection and loading based on file extension
2. **Feature Extraction**: Processes landmarks into hand shape, position, and lip features
3. **Neural Network**: Three-stream fusion encoder with CTC output
4. **Language Model**: KenLM-based beam search for French sentence correction
5. **Video Processing**: Generates subtitled output with synchronized audio

#### Generator Components
1. **Whisper Integration**: Automatic speech recognition for transcription
2. **MFA Alignment**: Montreal Forced Alignment for precise phoneme timing
3. **Cue Mapping**: Maps phonemes to hand shapes and positions using cued speech rules
4. **Hand Rendering**: Overlays realistic hand gestures onto video frames
5. **Synchronization**: Ensures perfect timing between speech and visual cues

### Model Architecture

#### Decoder Architecture
The decoder uses a three-stream fusion encoder:
- **Hand Shape Stream**: Processes hand landmark positions and geometric features
- **Hand Position Stream**: Analyzes hand movement and positioning
- **Lips Stream**: Extracts lip movement and facial features

#### Generator Architecture
The generator follows a multi-stage pipeline:
- **Audio Processing**: Whisper-based transcription and feature extraction
- **Phoneme Alignment**: MFA-based precise timing alignment
- **Cue Generation**: Rule-based mapping from phonemes to hand configurations
- **Video Rendering**: Real-time hand overlay with facial landmark tracking

### Processing Pipeline

#### Decoding Pipeline
1. **Video Input**: Load and process video frames
2. **Landmark Extraction**: Use MediaPipe to extract hand and face landmarks
3. **Feature Computation**: Calculate geometric and temporal features
4. **Model Inference**: Run CTC model to predict phonemes
5. **Language Correction**: Apply beam search with language models
6. **Subtitle Generation**: Create output video with French subtitles

#### Generation Pipeline
1. **Text Input**: Process French text for cued speech generation
2. **Audio Extraction**: Extract or use provided audio track
3. **Speech Recognition**: Use Whisper for accurate transcription
4. **Phoneme Alignment**: Apply MFA for precise timing
5. **Cue Mapping**: Map phonemes to hand shapes and positions
6. **Video Rendering**: Overlay hand cues with perfect synchronization


## License

This project is licensed under the MIT License - see the LICENSE file for details.


## TFLite Models Information

### Model Details

The decoder uses the latest MediaPipe float16 models for optimal accuracy:

| Model | Landmarks | Size | Precision | Complexity |
|-------|-----------|------|-----------|------------|
| **Face Landmarker** | 478 points | 3.6 MB | float16 | Standard |
| **Hand Landmarker** | 21 points/hand | 7.5 MB | float16 | Standard |
| **Pose Landmarker FULL** | 33 points | 9.0 MB | float16 | **Highest** |

### Model Sources

Models are automatically downloaded from official MediaPipe repositories:
- Face: [mediapipe-models/face_landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
- Hand: [mediapipe-models/hand_landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- Pose: [mediapipe-models/pose_landmarker_full](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)

### Advantages Over MediaPipe Holistic

1. **Higher Quality**: Float16 precision with latest model versions
2. **More Landmarks**: Face model provides 478 landmarks (vs 468 in older models)
3. **Better Pose Estimation**: FULL complexity model for more accurate body tracking
4. **Mobile-Ready**: Same `.task` files work seamlessly in Flutter mobile apps
5. **Future-Proof**: Direct access to latest MediaPipe models as they're updated

### Manual Model Management

If you need to download models separately:

```bash
# Download individual models
curl -L -o download/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

curl -L -o download/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

curl -L -o download/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Or use the provided script (legacy, for separate downloads):
```bash
bash download_tflite_models.sh
```

## Acknowledgments

- MediaPipe and MediaPipe Tasks API for landmark extraction
- Google for providing high-quality TFLite models
- PyTorch for deep learning framework
- KenLM for language modeling
- The cued speech research community

## Support

For questions and support:
- Contact: boubasow.pro@gmail.com
