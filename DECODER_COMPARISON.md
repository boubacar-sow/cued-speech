# Decoder Comparison: MediaPipe Holistic vs MediaPipe Tasks

## Configuration Differences

### MediaPipe Holistic (Original)
```python
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,      # VIDEO mode with tracking
    model_complexity=1,            # Medium complexity
    smooth_landmarks=True,         # Temporal smoothing enabled
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
```

### MediaPipe Tasks (New - IMAGE mode)
```python
# Currently using IMAGE mode (per-frame, no tracking)
options = vision.FaceLandmarkerOptions(
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## Key Differences

| Feature | MediaPipe Holistic | MediaPipe Tasks (Current) |
|---------|-------------------|---------------------------|
| **Running Mode** | VIDEO (tracking across frames) | IMAGE (per-frame, independent) |
| **Temporal Smoothing** | Yes (`smooth_landmarks=True`) | No (IMAGE mode) |
| **Tracking** | Yes (uses previous frame context) | No (IMAGE mode) |
| **Model Complexity** | 1 (medium) | Full (via model file) |
| **Face Landmarks** | 468 points | 478 points |
| **Detection Confidence** | 0.3 | 0.5 |

## Impact on Decoding Quality

The main difference affecting quality:

1. **Temporal Consistency**: Holistic tracks landmarks across frames, smoothing jitter
2. **Confidence Threshold**: Holistic uses 0.3, we're using 0.5 (may miss some detections)
3. **Landmark Count**: New model has 478 vs 468 face points (more detailed)

## Recommended Fixes

### Option 1: Lower Confidence Thresholds (Quick Fix)

Match Holistic's confidence levels:

```python
min_face_detection_confidence=0.3  # was 0.5
min_hand_detection_confidence=0.3  # was 0.5  
min_pose_detection_confidence=0.3  # was 0.5
```

### Option 2: Implement VIDEO Mode (Better Quality)

VIDEO mode requires callbacks and is more complex:

```python
def result_callback(result, output_image, timestamp_ms):
    # Handle results asynchronously
    pass

options = vision.FaceLandmarkerOptions(
    running_mode=RunningMode.VIDEO,  # Enable tracking
    num_faces=1,
    min_face_detection_confidence=0.3,
    result_callback=result_callback
)

# Process with timestamps
for frame_idx, frame in enumerate(frames):
    timestamp_ms = int(frame_idx * 1000 / fps)
    landmarker.detect_async(mp_image, timestamp_ms)
```

### Option 3: Add Manual Temporal Smoothing

Implement smoothing filter on landmarks:

```python
# Simple exponential moving average
smoothed_landmark = alpha * current_landmark + (1 - alpha) * previous_landmark
```

## Testing Strategy

1. **Test with lower confidence** (0.3 instead of 0.5)
2. **Compare landmark trajectories** between Holistic and Tasks
3. **If still differs, implement VIDEO mode** for temporal consistency
4. **If still differs, add manual smoothing**

## Current Status

✅ Using latest float16 models (highest quality)
✅ Using MediaPipe Tasks API natively (.task files)
⚠️ Using IMAGE mode (no temporal tracking)
⚠️ Using higher confidence threshold (0.5 vs 0.3)

## Next Steps

Run test to see if updated models + lower confidence fixes the issue:

```bash
# Edit decoder_tflite.py to set min_*_confidence=0.3
pixi run cued-speech decode --video_path download/test_decode.mp4
```

If quality is still off, implement VIDEO mode for proper temporal tracking.

