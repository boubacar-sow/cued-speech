# Quick Start Guide

Get the C++ Cued Speech Decoder running in 30 minutes.

## Prerequisites

- Linux, macOS, or Windows with WSL
- C++17 compiler (gcc >= 7)
- CMake >= 3.16
- Git

## 1. Install Dependencies (Ubuntu/Debian)

```bash
# Build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Required libraries
sudo apt-get install -y libboost-all-dev libbz2-dev liblzma-dev zlib1g-dev
```

## 2. Build and Install KenLM

```bash
cd /tmp
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

## 3. Build and Install flashlight-text

```bash
cd /tmp
git clone https://github.com/flashlight/text.git
cd text
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_TEXT_USE_KENLM=ON
make -j$(nproc)
sudo make install
```

## 4. Build Cued Speech Decoder

```bash
cd /path/to/cued_speech/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Optional: install system-wide
sudo make install
```

## 5. Test the Build

```bash
# Check library was created
ls -lh libcued_speech_decoder.*

# Build and run example (if you have model files)
cd ..
gcc -o example example_usage.c -L./build -lcued_speech_decoder -Wl,-rpath,./build
./example lexicon.txt tokens.txt lm.bin
```

## Troubleshooting

### "flashlight-text not found"

```bash
# Check if installed
ls /usr/local/lib/libflashlight-text.*

# If not found, ensure install succeeded
cd /tmp/text/build
sudo make install

# Update library cache
sudo ldconfig
```

### "KenLM not found"

```bash
# Check installation
ls /usr/local/lib/libkenlm.*

# Manually specify path
cmake .. -DKENLM_INCLUDE_DIR=/usr/local/include \
         -DKENLM_LIBRARY=/usr/local/lib/libkenlm.so
```

### Runtime library errors

```bash
# Check dependencies
ldd build/libcued_speech_decoder.so

# Add to library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

## Next Steps

1. **Complete the implementation** - See `IMPLEMENTATION_NOTES.md` for TODOs
2. **Test against Python** - Validate outputs match
3. **Cross-compile for mobile** - See `README.md` for Android/iOS builds
4. **Integrate with Flutter** - See `FLUTTER_INTEGRATION.md`

## Quick Commands Reference

```bash
# Build
cd build && make -j$(nproc)

# Clean rebuild
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)

# Run tests (when implemented)
cd build && ctest --output-on-failure

# Install
sudo make install

# Uninstall
sudo rm /usr/local/lib/libcued_speech_decoder.*
sudo rm -rf /usr/local/include/cued_speech/
```

## Docker Alternative (Recommended for Reproducibility)

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libboost-all-dev libbz2-dev liblzma-dev zlib1g-dev

# Install KenLM
RUN cd /tmp && git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install

# Install flashlight-text
RUN cd /tmp && git clone https://github.com/flashlight/text.git && \
    cd text && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_TEXT_USE_KENLM=ON && \
    make -j$(nproc) && make install

WORKDIR /workspace
```

Build and use:

```bash
docker build -t cued-speech-builder .
docker run -v $(pwd):/workspace -it cued-speech-builder

# Inside container
cd cpp && mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Support

For issues:
1. Check `IMPLEMENTATION_NOTES.md` for known limitations
2. Review build logs for specific errors
3. Ensure all dependencies are correctly installed
4. Try Docker approach for clean environment

