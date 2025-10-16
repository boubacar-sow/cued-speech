#!/bin/bash
# Script to download MediaPipe TFLite models for cued speech decoder

set -e  # Exit on error

echo "========================================"
echo "Downloading TFLite Models"
echo "========================================"

# Create directory for models
MODELS_DIR="tflite_models"
mkdir -p "$MODELS_DIR"

echo ""
echo "Created directory: $MODELS_DIR"
echo ""

# Function to download file with retry
download_file() {
    local url=$1
    local output=$2
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo "Downloading: $(basename $output)"
        if curl -L -f -o "$output" "$url"; then
            echo "✓ Successfully downloaded: $(basename $output)"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "Failed. Retrying ($retry/$max_retries)..."
                sleep 2
            fi
        fi
    done
    
    echo "✗ Failed to download after $max_retries attempts: $(basename $output)"
    return 1
}

# Download Face Mesh Model (float16 latest for best quality)
echo "1. Downloading Face Landmark Model (float16, latest)..."
FACE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_OUTPUT="$MODELS_DIR/face_landmarker.task"

if download_file "$FACE_URL" "$FACE_OUTPUT"; then
    echo "   ✓ Face model saved to: $FACE_OUTPUT"
else
    echo "   ✗ Warning: Could not download face model from official source"
    echo "   Trying fallback URL..."
    FACE_URL_FALLBACK="https://storage.googleapis.com/mediapipe-assets/face_landmarker.task"
    if download_file "$FACE_URL_FALLBACK" "$FACE_OUTPUT"; then
        echo "   ✓ Face model saved to: $FACE_OUTPUT (fallback)"
    else
        echo "   You may need to download manually from:"
        echo "   https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
    fi
fi

echo ""

# Download Hand Landmark Model (float16 latest for best quality)
echo "2. Downloading Hand Landmark Model (float16, latest)..."
HAND_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_OUTPUT="$MODELS_DIR/hand_landmarker.task"

if download_file "$HAND_URL" "$HAND_OUTPUT"; then
    echo "   ✓ Hand model saved to: $HAND_OUTPUT"
else
    echo "   ✗ Warning: Could not download hand model from official source"
    echo "   Trying fallback URL..."
    HAND_URL_FALLBACK="https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    if download_file "$HAND_URL_FALLBACK" "$HAND_OUTPUT"; then
        echo "   ✓ Hand model saved to: $HAND_OUTPUT (fallback)"
    else
        echo "   You may need to download manually from:"
        echo "   https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker"
    fi
fi

echo ""

# Download Pose Landmark Model FULL (float16 latest for best quality)
echo "3. Downloading Pose Landmark Model FULL (float16, latest - highest complexity)..."
POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
POSE_OUTPUT="$MODELS_DIR/pose_landmarker_full.task"

if download_file "$POSE_URL" "$POSE_OUTPUT"; then
    echo "   ✓ Pose model (FULL) saved to: $POSE_OUTPUT"
else
    echo "   ✗ Warning: Could not download pose FULL model from official source"
    echo "   Trying standard pose model as fallback..."
    POSE_URL_FALLBACK="https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task"
    POSE_OUTPUT_FALLBACK="$MODELS_DIR/pose_landmarker.task"
    if download_file "$POSE_URL_FALLBACK" "$POSE_OUTPUT_FALLBACK"; then
        echo "   ✓ Pose model (standard) saved to: $POSE_OUTPUT_FALLBACK (fallback)"
    else
        echo "   You may need to download manually from:"
        echo "   https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker"
    fi
fi

echo ""
echo "========================================"
echo "Download Summary"
echo "========================================"

# Check which files were successfully downloaded
TOTAL=0
SUCCESS=0

for file in "$MODELS_DIR"/*.task; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "✓ $(basename $file) - $SIZE"
        SUCCESS=$((SUCCESS + 1))
    fi
    TOTAL=$((TOTAL + 1))
done

echo ""
echo "Downloaded $SUCCESS out of 3 models successfully"
echo ""

if [ $SUCCESS -eq 3 ]; then
    echo "All models downloaded successfully! ✓"
    echo ""
    echo "You can now use these models with decoder_tflite.py:"
    echo "  - Face model: $FACE_OUTPUT"
    echo "  - Hand model: $HAND_OUTPUT"
    echo "  - Pose model: $POSE_OUTPUT"
else
    echo "Some models failed to download."
    echo ""
    echo "Alternative download methods:"
    echo ""
    echo "1. Manual download from MediaPipe Solutions:"
    echo "   - Face: https://developers.google.com/mediapipe/solutions/vision/face_landmarker"
    echo "   - Hand: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker"
    echo "   - Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker"
    echo ""
    echo "2. Download from Hugging Face (Qualcomm optimized for Pose):"
    echo "   - Pose: https://huggingface.co/qualcomm/MediaPipe-Pose-Estimation"
    echo ""
    echo "3. Using Python script (alternative):"
    echo "   python -c 'import mediapipe as mp; mp.solutions.pose'"
fi

echo ""
echo "For more information, see TFLITE_MODELS_GUIDE.md"
echo "========================================"

echo ""
echo "IMPORTANT: The files downloaded by this script are .task files."
echo ""
echo "✅ GOOD NEWS: decoder_tflite.py now supports .task files natively!"
echo "   - .task files will be loaded using the MediaPipe Tasks API"
echo "   - No extraction or conversion needed"
echo "   - Just use the .task files directly with --face_tflite, --hand_tflite, --pose_tflite"
echo ""
echo "Alternative: If you prefer to use raw .tflite models:"
echo "   - Provide .tflite files instead of .task files"
echo "   - These will be loaded using the TFLite Interpreter"
echo "   - Example: hand_landmark_full.tflite, face_landmark.tflite, etc."
echo ""
echo "The decoder will automatically detect the file type and use the appropriate API."

