# Model Quality Improvements

## Summary

Upgraded to the latest MediaPipe float16 models for improved landmark detection quality, which should improve decoding accuracy.

## Changes Made

### 1. Updated Model Downloads

**Previous models (from `mediapipe-assets`):**
- Face: 1.34 MB (older version)
- Hand: 7.46 MB 
- Pose: 5.51 MB (standard)

**New models (from `mediapipe-models`, float16/latest):**
- Face: **3.58 MB** (float16 latest) - 2.7x larger, more accurate
- Hand: **7.46 MB** (float16 latest) - same size, latest version
- Pose: **8.96 MB** (FULL complexity) - 1.6x larger, highest quality

Sources (as per [MediaPipe Solutions guide](https://ai.google.dev/edge/mediapipe/solutions/guide)):
- Face: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- Hand: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- Pose: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`

### 2. Updated Model Configuration

Added quality/confidence thresholds to MediaPipe Tasks API:

```python
# Face landmarker
min_face_detection_confidence=0.5
min_face_presence_confidence=0.5
min_tracking_confidence=0.5

# Hand landmarker  
min_hand_detection_confidence=0.5
min_hand_presence_confidence=0.5
min_tracking_confidence=0.5

# Pose landmarker
min_pose_detection_confidence=0.5
min_pose_presence_confidence=0.5
min_tracking_confidence=0.5
```

These match standard MediaPipe quality settings for production use.

### 3. Updated CLI Defaults

Changed default pose model path from `pose_landmarker.task` (standard) to `pose_landmarker_full.task` (highest complexity) for best quality.

### 4. Face Landmark Count

The new face model outputs **478 landmarks** (vs 468 in older models), providing more detailed facial features including:
- Additional eye region landmarks
- More precise lip contours
- Enhanced cheek/face boundary points

## Expected Improvements

1. **Better Face Detection**: More landmarks = more precise lip reading
2. **Improved Pose Estimation**: FULL model provides better body/shoulder tracking
3. **Consistent Hand Detection**: Latest hand model with proper confidence thresholds

## Testing

Run the test script:

```bash
bash test_new_models.sh
```

Or manually:

```bash
pixi run cued-speech decode \
  --video_path download/test_decode.mp4 \
  --face_tflite tflite_models/face_landmarker.task \
  --hand_tflite tflite_models/hand_landmarker.task \
  --pose_tflite tflite_models/pose_landmarker_full.task \
  --output_path output/decoder/decoded_video_new_models.mp4
```

## Troubleshooting

If decoding quality is still different from MediaPipe Holistic:

1. **Check landmark preprocessing**: Verify normalization/scaling matches between implementations
2. **Feature extraction**: Ensure the same landmark indices are used (LIP_INDICES, HAND_INDICES, etc.)
3. **Temporal smoothing**: MediaPipe Holistic may apply temporal filtering; Tasks API in IMAGE mode does not
4. **Model versions**: Verify MediaPipe Holistic uses same underlying models

## Model Version Comparison

| Component | MediaPipe Holistic | MediaPipe Tasks (New) |
|-----------|-------------------|----------------------|
| Face | Face Mesh (468 landmarks) | Face Landmarker float16 (478 landmarks) |
| Hand | Hand Landmarker | Hand Landmarker float16 latest |
| Pose | Pose Landmarker | Pose Landmarker FULL float16 |
| Running Mode | VIDEO (with tracking) | IMAGE (per-frame) |
| Temporal Filtering | Yes (built-in) | No (per-frame) |

## Next Steps if Quality Still Differs

1. **Switch to VIDEO mode**: Change `RunningMode.IMAGE` to `RunningMode.VIDEO` for temporal consistency
2. **Add temporal smoothing**: Implement landmark smoothing across frames
3. **Verify coordinate systems**: Check if coordinate normalization differs
4. **Compare MediaPipe Holistic config**: Match exact confidence thresholds used by Holistic

## Files Modified

- `download_tflite_models.sh` - Updated URLs to latest float16 models
- `src/cued_speech/decoder_tflite.py` - Added confidence thresholds
- `src/cued_speech/cli.py` - Updated default pose model to FULL variant
- Added `test_new_models.sh` - Testing script

## References

- [MediaPipe Solutions Guide](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [Face Landmarker Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [Hand Landmarker Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Pose Landmarker Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)

