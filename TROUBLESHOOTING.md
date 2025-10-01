# Troubleshooting Guide

This guide helps you resolve common issues when installing and using the Cued Speech package.

## Installation Issues

### NumPy/PyTorch Compatibility Error ⚠️ **COMMON ISSUE**

**Error Message:**
```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

**Why this happens:**
- You have NumPy 2.x installed
- PyTorch was compiled against NumPy 1.x
- This creates a compatibility conflict

**Quick Fix:**
```bash
# Downgrade NumPy to compatible version
pip install "numpy>=1.24,<2.0"

# Reinstall the package
pip install --upgrade cued-speech
```

**Alternative Solutions:**
1. **Use a virtual environment:**
   ```bash
   python -m venv cued-speech-env
   source cued-speech-env/bin/activate  # On Windows: cued-speech-env\Scripts\activate
   pip install "numpy>=1.24,<2.0"
   pip install cued-speech
   ```

2. **Use conda:**
   ```bash
   conda create -n cued-speech python=3.11
   conda activate cued-speech
   conda install "numpy>=1.24,<2.0"
   pip install cued-speech
   ```

3. **Upgrade PyTorch** (if available):
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

### SSL Certificate Issues

**Error Message:**
```
❌ Error downloading data: Failed to download data: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**Solution:**
The package now handles SSL issues automatically, but if you still have problems:

1. **Update the package** to the latest version:
   ```bash
   pip install --upgrade cued-speech
   ```

2. **Install certificates** (macOS):
   ```bash
   # Navigate to your Python directory
   cd /Applications/Python\ 3.11/
   ./Install\ Certificates.command
   ```

3. **Manual download**: If automatic download fails, you can manually download the data:
   ```bash
   # Download from the GitHub release
   curl -L -o download.zip https://github.com/boubacar-sow/CuedSpeechRecognition/releases/download/cuedspeech/download.zip
   unzip download.zip
   ```

### Missing Dependencies

**Error Message:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
Install missing dependencies:

```bash
# Install PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install mediapipe opencv-python pandas
```

## Usage Issues

### Data Download Problems

**Problem:** Data files won't download automatically.

**Solutions:**

1. **Check internet connection**:
   ```bash
   ping github.com
   ```

2. **Try manual download**:
   ```bash
   cued-speech download-data --force
   ```

3. **Check available data**:
   ```bash
   cued-speech list-data
   ```

4. **Clean and retry**:
   ```bash
   cued-speech cleanup-data --confirm
   cued-speech download-data
   ```

### File Path Issues

**Problem:** Can't find model files or data.

**Solutions:**

1. **Check default paths**:
   ```bash
   # The package looks for files in these locations:
   # - ./download/ (current working directory)
   # - package_data/download/ (package installation directory)
   ```

2. **Specify custom paths**:
   ```bash
   cued-speech decode \
     --model_path /path/to/your/model.pt \
     --vocab_path /path/to/your/vocab.csv
   ```

3. **Verify file existence**:
   ```bash
   ls -la download/
   ```

### Memory Issues

**Problem:** Out of memory errors during processing.

**Solutions:**

1. **Reduce batch size** (if applicable)
2. **Use CPU instead of GPU**:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   cued-speech decode
   ```
3. **Close other applications** to free memory
4. **Process smaller video files** first

### Video Processing Issues

**Problem:** Video won't process or errors occur.

**Solutions:**

1. **Check video format**:
   ```bash
   # Ensure video is MP4 format
   ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
   ```

2. **Install FFmpeg**:
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

3. **Check video file integrity**:
   ```bash
   ffprobe your_video.mp4
   ```

## Platform-Specific Issues

### macOS Issues

**Problem:** MediaPipe or OpenCV not working.

**Solution:**
```bash
# Install with specific versions for macOS
pip install mediapipe==0.10.14
pip install opencv-python==4.12.0.88
```

### Windows Issues

**Problem:** Path issues or encoding problems.

**Solution:**
```bash
# Use forward slashes or raw strings for paths
cued-speech decode --video_path "C:/path/to/video.mp4"

# Or use relative paths
cued-speech decode --video_path "./video.mp4"
```

### Linux Issues

**Problem:** Missing system libraries.

**Solution:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# CentOS/RHEL
sudo yum install mesa-libGL libXext libXrender
```

## Performance Issues

### Slow Processing

**Solutions:**

1. **Use GPU acceleration** (if available):
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Reduce video resolution**:
   ```bash
   ffmpeg -i input.mp4 -vf scale=640:480 output.mp4
   ```

3. **Process shorter videos** for testing

### High CPU Usage

**Solutions:**

1. **Limit CPU cores**:
   ```bash
   export OMP_NUM_THREADS=4
   cued-speech decode
   ```

2. **Use lower quality settings** (if available)

## Getting Help

If you're still having issues:

1. **Check the logs** for detailed error messages
2. **Try the test script**:
   ```bash
   python test_data_manager.py
   ```
3. **Create a minimal example** to reproduce the issue
4. **Check GitHub issues** for similar problems
5. **Open a new issue** with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue
   - Package version: `pip show cued-speech`

## Common Commands

```bash
# Check package version
pip show cued-speech

# Check Python version
python --version

# Check NumPy version
python -c "import numpy; print(numpy.__version__)"

# List installed packages
pip list

# Test data manager
python test_data_manager.py

# Clean environment (if needed)
pip uninstall cued-speech
pip install cued-speech
``` 