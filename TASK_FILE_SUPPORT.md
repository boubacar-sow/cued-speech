# MediaPipe .task File Support

## Overview

The `decoder_tflite.py` module now natively supports MediaPipe `.task` files! You no longer need to extract or convert them to use landmark detection models.

## What Changed

### Automatic Detection
The decoder automatically detects your model file type and uses the appropriate API:

- **`.task` files** → Loaded with MediaPipe Tasks API
- **`.tflite` files** → Loaded with TFLite Interpreter

### No Manual Configuration Needed
Just provide the model path, and the decoder handles everything:

```bash
# Using .task files (MediaPipe Tasks API)
cued-speech decode \
  --face_tflite tflite_models/face_landmarker.task \
  --hand_tflite tflite_models/hand_landmarker.task \
  --pose_tflite tflite_models/pose_landmarker.task \
  --video_path your_video.mp4

# Using .tflite files (TFLite Interpreter)
cued-speech decode \
  --face_tflite models/face_landmark.tflite \
  --hand_tflite models/hand_landmark_full.tflite \
  --pose_tflite models/pose_landmark.tflite \
  --video_path your_video.mp4
```

## Architecture

### New Components

1. **`MediaPipeTasksWrapper`** (decoder_tflite.py:318-398)
   - Wraps MediaPipe Tasks API landmarkers
   - Handles FaceLandmarker, HandLandmarker, PoseLandmarker
   - Converts MediaPipe native output to our custom format

2. **Enhanced `MediaPipeStyleLandmarkExtractor`**
   - Auto-detects file extension (.task vs .tflite)
   - Routes to appropriate wrapper (MediaPipe Tasks or TFLite Interpreter)
   - Unified interface regardless of backend

3. **Output Format Conversion**
   - MediaPipe Tasks returns native landmark objects
   - TFLite Interpreter returns numpy arrays
   - Both are converted to the same `Landmarks` format for downstream processing

### Detection Flow

```
User provides model path
    ↓
Check file extension
    ↓
┌─────────────────────────┬──────────────────────────┐
│  .task extension        │  .tflite extension       │
│  ↓                      │  ↓                       │
│  MediaPipeTasksWrapper  │  TFLiteModelWrapper      │
│  ↓                      │  ↓                       │
│  MediaPipe Tasks API    │  TFLite Interpreter      │
└─────────────────────────┴──────────────────────────┘
    ↓
Convert to unified Landmarks format
    ↓
Pass to feature extractor (existing code)
```

## Installation Requirements

### For .task Files
```bash
pip install mediapipe
```

The package already includes MediaPipe in its dependencies, so this should be available.

### For .tflite Files
```bash
pip install tflite-runtime
```

## Model Downloads

Run the provided script to download MediaPipe .task models:

```bash
bash download_tflite_models.sh
```

This will download:
- `face_landmarker.task` (468 face landmarks)
- `hand_landmarker.task` (21 hand landmarks per hand)
- `pose_landmarker.task` (33 pose landmarks)

## Benefits of .task Files

1. **Official MediaPipe Format**: No need to extract or convert
2. **Built-in Pre/Post-Processing**: Optimized for performance
3. **Easier to Obtain**: Available from official MediaPipe sources
4. **Well-Tested**: Same models used in MediaPipe examples

## Hand Detection for Cued Speech

The hand landmarker detects both hands and automatically selects the right hand (preferred for cued speech):

```python
# In MediaPipeTasksWrapper.inference() for hand models:
# - Detects up to 2 hands
# - Identifies handedness (Left/Right)
# - Prefers right hand for cued speech
# - Falls back to first detected hand if right not found
```

## Error Handling

If a model file fails to load, you'll see clear error messages:

```
ValueError: Model provided appears to be a MediaPipe .task file, which cannot be loaded 
by the TFLite Interpreter. Provide a raw .tflite model instead...
```

Or:

```
RuntimeError: MediaPipe Tasks API is required for .task files but is not available. 
Install with: pip install mediapipe
```

## Fallback to MediaPipe Holistic

If TFLite/Tasks decoding fails for any reason, the CLI automatically falls back to the original MediaPipe Holistic decoder:

```bash
⚠️ TFLite decoding unavailable: <error message>. Falling back to MediaPipe decoder.
🧠 Using MediaPipe Holistic-based landmark detection
```

## Testing Your Setup

1. **Download models:**
   ```bash
   bash download_tflite_models.sh
   ```

2. **Run decoding:**
   ```bash
   cued-speech decode \
     --video_path download/test_decode.mp4 \
     --face_tflite tflite_models/face_landmarker.task \
     --hand_tflite tflite_models/hand_landmarker.task \
     --pose_tflite tflite_models/pose_landmarker.task
   ```

3. **Check output:**
   - Should see: "🧠 Using MediaPipe Tasks API for landmark detection (.task files)"
   - Should see model loading messages with model type and path
   - Video should decode successfully

## Implementation Details

### MediaPipe Tasks Output Format

```python
# Face detection result
{
    'face_landmarks': [
        [NormalizedLandmark(x=..., y=..., z=...), ...]  # 468 landmarks
    ]
}

# Hand detection result
{
    'hand_landmarks': [
        [NormalizedLandmark(x=..., y=..., z=...), ...],  # 21 landmarks per hand
        ...
    ],
    'handedness': [
        [Category(category_name='Right', score=...)],
        ...
    ]
}

# Pose detection result
{
    'pose_landmarks': [
        [NormalizedLandmark(x=..., y=..., z=...), ...]  # 33 landmarks
    ]
}
```

### TFLite Output Format

```python
# Raw numpy arrays
{
    'output_0': np.ndarray(shape=(1, num_landmarks, 3))  # (batch, landmarks, xyz)
}
```

Both formats are converted to:

```python
class Landmarks:
    landmark: List[Landmark]  # List of Landmark(x, y, z) objects
```

## Future Enhancements

Potential improvements:

1. **Video Mode Support**: Currently using IMAGE mode; could optimize for VIDEO mode
2. **Batch Processing**: Process multiple frames in parallel
3. **Model Caching**: Cache loaded models between runs
4. **Performance Metrics**: Add timing information for each model
5. **Dynamic Model Selection**: Auto-download missing models

## Troubleshooting

### Issue: "MediaPipe Tasks API not available"
**Solution:** Install MediaPipe: `pip install mediapipe`

### Issue: "TFLite runtime not available"
**Solution:** Install TFLite runtime: `pip install tflite-runtime`

### Issue: Model file not found
**Solution:** Run `bash download_tflite_models.sh` or check your paths

### Issue: Poor detection quality
**Solution:** 
- Ensure good lighting in video
- Check that face/hand are clearly visible
- Verify video resolution is adequate (>= 640x480)

## Questions?

For more information:
- See `FLUTTER_INTEGRATION.md` for mobile deployment
- See `download_tflite_models.sh` for model sources
- Check the code: `src/cued_speech/decoder_tflite.py`

