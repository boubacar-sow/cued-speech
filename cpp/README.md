# Cued Speech Decoder - C++ Implementation

This is a C++ port of the Python `decoder_tflite.py` for use in mobile applications via Flutter FFI.

## Features

- **CTC Beam Search Decoding** using [flashlight-text](https://github.com/flashlight/text)
- **Language Model Integration** with [KenLM](https://github.com/kpu/kenlm)
- **Streaming Decoding** with overlap-save windowing (matches Python behavior)
- **C API** for easy FFI integration with Flutter/Dart
- **Feature Extraction** from landmarks (hand shape, hand position, lips)
- **Phoneme Correction** with homophone selection

## Dependencies

### Required

1. **flashlight-text** (MIT License)
   - CTC decoder, lexicon, dictionary utilities
   - https://github.com/flashlight/text

2. **KenLM** (LGPL-2.1)
   - N-gram language model
   - https://github.com/kpu/kenlm

3. **TensorFlow Lite C++**
   - Three-stream feature encoder exported as TFLite
   - https://www.tensorflow.org/lite/guide/build_c

4. **OpenCV** (core, imgproc, videoio)
   - Video IO and subtitle overlay
   - https://opencv.org/

5. **CMake** >= 3.16
6. **C++17 compiler** (gcc >= 7, clang >= 5, MSVC >= 2019)

### Optional

- **TensorFlow Lite** (if integrating TFLite inference in C++)
- **Google Test** (for tests)

## Building from Source

### 1. Install Dependencies

#### Ubuntu/Debian

```bash
# Install build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake git

# Install Boost (required by KenLM)
sudo apt-get install -y libboost-all-dev

# Install compression libraries (for KenLM)
sudo apt-get install -y libbz2-dev liblzma-dev zlib1g-dev
```

#### macOS

```bash
brew install cmake boost libbz2 xz zlib
```

#### Windows

- Install Visual Studio 2019 or later with C++ support
- Install CMake from https://cmake.org/download/
- Install vcpkg and use it to install dependencies:
  ```powershell
  vcpkg install boost:x64-windows
  ```

### 2. Build and Install KenLM

```bash
cd /tmp
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

### 3. Build and Install flashlight-text

```bash
cd /tmp
git clone https://github.com/flashlight/text.git
cd text
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFL_TEXT_USE_KENLM=ON
make -j$(nproc)
sudo make install
```

### 4. Build Cued Speech Decoder

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

#### Build Options

- `-DBUILD_SHARED_LIBS=ON/OFF` - Build shared or static library (default: ON)
- `-DBUILD_EXAMPLES=ON/OFF` - Build example programs (default: OFF)
- `-DBUILD_TESTS=ON/OFF` - Build tests (default: OFF)

## Cross-Compilation for Mobile

### Android (NDK)

```bash
cd cpp
mkdir build-android && cd build-android

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON

make -j$(nproc)
```

Output: `libcued_speech_decoder.so` for Android

### iOS

```bash
cd cpp
mkdir build-ios && cd build-ios

cmake .. \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF

make -j$(nproc)
```

Output: `libcued_speech_decoder.a` for iOS

**Note:** For iOS, you may need to address LGPL compliance for KenLM (see main README).

## C API Usage

### Basic Decoding

```c
#include <cued_speech/decoder_c_api.h>

// Create decoder configuration
DecoderConfig config = decoder_config_default();
config.lexicon_path = "lexicon.txt";
config.tokens_path = "tokens.txt";
config.lm_path = "kenlm_model.bin";
config.beam_size = 40;
config.lm_weight = 3.23f;

// Initialize decoder
DecoderHandle decoder = decoder_create(&config);
if (!decoder) {
    fprintf(stderr, "Error: %s\n", decoder_get_last_error());
    return 1;
}

// Decode logits [T x V]
int T = 100;  // time steps
int V = decoder_get_vocab_size(decoder);
float* logits = /* your logits */;

int num_results = 0;
Hypothesis* results = decoder_decode(decoder, logits, T, V, &num_results);

if (results) {
    // Print best hypothesis
    printf("Score: %.3f\n", results[0].score);
    printf("Tokens: ");
    for (int i = 0; i < results[0].words_length; i++) {
        printf("%s ", results[0].words[i]);
    }
    printf("\n");
    
    decoder_free_hypotheses(results, num_results);
}

// Clean up
decoder_destroy(decoder);
```

### Streaming Decoding

```c
// Create streaming session
StreamHandle stream = stream_create(decoder);

// Load TFLite sequence model (required for streaming inference)
if (!stream_load_tflite_model(stream, "model.tflite")) {
    fprintf(stderr, "Failed to load TFLite model: %s\n", decoder_get_last_error());
}

// Process frames
for (int i = 0; i < num_frames; i++) {
    float features[33] = /* extract from landmarks */;
    
    if (stream_push_frame(stream, features)) {
        RecognitionResult* result = stream_process_window(stream);
        
        if (result && result->phonemes_length > 0) {
            printf("Frame %d: ", result->frame_number);
            for (int j = 0; j < result->phonemes_length; j++) {
                printf("%s ", result->phonemes[j]);
            }
            printf("\n");
        }
        
        stream_free_result(result);
    }
}

RecognitionResult* final_result = stream_finalize(stream);
stream_free_result(final_result);
stream_destroy(stream);
```

## Flutter FFI Integration

### 1. Copy Library to Flutter Project

```bash
# Android
cp build-android/libcued_speech_decoder.so flutter_app/android/app/src/main/jniLibs/arm64-v8a/

# iOS
cp build-ios/libcued_speech_decoder.a flutter_app/ios/Frameworks/
```

### 2. Create Dart Bindings

```dart
// lib/decoder_bindings.dart
import 'dart:ffi';
import 'package:ffi/ffi.dart';

// Load library
final DynamicLibrary nativeLib = Platform.isAndroid
    ? DynamicLibrary.open('libcued_speech_decoder.so')
    : DynamicLibrary.process();

// C structures
class DecoderConfig extends Struct {
  external Pointer<Utf8> lexicon_path;
  external Pointer<Utf8> tokens_path;
  external Pointer<Utf8> lm_path;
  external Pointer<Utf8> lm_dict_path;
  
  @Int32()
  external int nbest;
  @Int32()
  external int beam_size;
  @Int32()
  external int beam_size_token;
  @Float()
  external double beam_threshold;
  @Float()
  external double lm_weight;
  @Float()
  external double word_score;
  @Float()
  external double unk_score;
  @Float()
  external double sil_score;
  @Bool()
  external bool log_add;
  
  external Pointer<Utf8> blank_token;
  external Pointer<Utf8> sil_token;
  external Pointer<Utf8> unk_word;
}

// Function signatures
typedef DecoderCreateNative = Pointer<Void> Function(Pointer<DecoderConfig>);
typedef DecoderCreate = Pointer<Void> Function(Pointer<DecoderConfig>);

typedef DecoderDestroyNative = Void Function(Pointer<Void>);
typedef DecoderDestroy = void Function(Pointer<Void>);

typedef StreamCreateNative = Pointer<Void> Function(Pointer<Void>);
typedef StreamCreate = Pointer<Void> Function(Pointer<Void>);

typedef StreamPushFrameNative = Bool Function(Pointer<Void>, Pointer<Float>);
typedef StreamPushFrame = bool Function(Pointer<Void>, Pointer<Float>);

// Lookup functions
final decoderCreate = nativeLib
    .lookup<NativeFunction<DecoderCreateNative>>('decoder_create')
    .asFunction<DecoderCreate>();

final decoderDestroy = nativeLib
    .lookup<NativeFunction<DecoderDestroyNative>>('decoder_destroy')
    .asFunction<DecoderDestroy>();

final streamCreate = nativeLib
    .lookup<NativeFunction<StreamCreateNative>>('stream_create')
    .asFunction<StreamCreate>();

final streamPushFrame = nativeLib
    .lookup<NativeFunction<StreamPushFrameNative>>('stream_push_frame')
    .asFunction<StreamPushFrame>();
```

### 3. Use in Flutter

```dart
// lib/decoder.dart
import 'decoder_bindings.dart';
import 'dart:ffi';
import 'package:ffi/ffi.dart';

class CuedSpeechDecoder {
  Pointer<Void>? _decoderHandle;
  Pointer<Void>? _streamHandle;
  
  Future<void> initialize(String lexiconPath, String tokensPath, String lmPath) async {
    final config = calloc<DecoderConfig>();
    config.ref.lexicon_path = lexiconPath.toNativeUtf8();
    config.ref.tokens_path = tokensPath.toNativeUtf8();
    config.ref.lm_path = lmPath.toNativeUtf8();
    config.ref.beam_size = 40;
    config.ref.lm_weight = 3.23;
    config.ref.blank_token = '<BLANK>'.toNativeUtf8();
    config.ref.sil_token = '_'.toNativeUtf8();
    config.ref.unk_word = '<UNK>'.toNativeUtf8();
    
    _decoderHandle = decoderCreate(config);
    
    if (_decoderHandle == nullptr) {
      throw Exception('Failed to create decoder');
    }
    
    _streamHandle = streamCreate(_decoderHandle!);
  }
  
  bool pushFrame(List<double> features) {
    if (_streamHandle == null || features.length != 33) {
      return false;
    }
    
    final featuresPtr = calloc<Float>(33);
    for (int i = 0; i < 33; i++) {
      featuresPtr[i] = features[i];
    }
    
    final ready = streamPushFrame(_streamHandle!, featuresPtr);
    
    calloc.free(featuresPtr);
    return ready;
  }
  
  void dispose() {
    if (_streamHandle != null) {
      // stream_destroy(_streamHandle!);
    }
    if (_decoderHandle != null) {
      decoderDestroy(_decoderHandle!);
    }
  }
}
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Flutter/Dart Application        │
└─────────────────┬───────────────────────┘
                  │ dart:ffi
┌─────────────────▼───────────────────────┐
│         C API (decoder_c_api.h)         │
│  - decoder_create/destroy               │
│  - stream_create/push_frame/process     │
│  - corrector_correct                    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       C++ Core (decoder.h/cpp)          │
│  - CTCDecoder                           │
│  - WindowProcessor                      │
│  - FeatureExtractor                     │
│  - SentenceCorrector                    │
└───┬─────────────┬───────────────────────┘
    │             │
    ▼             ▼
┌───────────┐ ┌──────────┐
│flashlight-│ │  KenLM   │
│   text    │ │          │
└───────────┘ └──────────┘
```

## Performance Considerations

1. **Memory Mapping**: KenLM models are memory-mapped for fast loading
2. **Windowing**: Overlap-save approach reduces latency for streaming
3. **Beam Search**: Configurable beam size balances accuracy vs speed
4. **No Copies**: FFI uses pointers to avoid unnecessary data copies
5. **Threading**: Can run decoding in separate thread/isolate in Dart

## Testing

Build and run tests:

```bash
cd cpp/build
cmake .. -DBUILD_TESTS=ON
make
ctest --output-on-failure
```

## Troubleshooting

### Library not found at runtime

**Android**: Ensure `.so` is in `jniLibs/<ABI>/`
**iOS**: Add framework to "Embedded Binaries" in Xcode

### Symbol not found

Check symbols in library:
```bash
nm -D libcued_speech_decoder.so | grep decoder_create
```

### LGPL Compliance (iOS)

For iOS App Store distribution with KenLM (LGPL):
- Provide relinkable object files, OR
- Replace KenLM with a permissive alternative (neural LM in TFLite)

## License

- This C++ code: Match your project license
- flashlight-text: MIT
- KenLM: LGPL-2.1 (see compliance notes above)

## References

- [flashlight-text GitHub](https://github.com/flashlight/text)
- [KenLM GitHub](https://github.com/kpu/kenlm)
- [Flutter FFI Documentation](https://dart.dev/guides/libraries/c-interop)
- [TorchAudio CTC Decoder](https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html)

