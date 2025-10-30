# Flutter Integration Guide

This guide shows how to integrate the C++ Cued Speech Decoder into your Flutter mobile app.

## Overview

The integration follows this flow:

```
Flutter/Dart App
    ↓ (dart:ffi)
C API (decoder_c_api.h)
    ↓
C++ Core (decoder.cpp)
    ↓
flashlight-text + KenLM
```

## Step 1: Build Native Libraries

### For Android (arm64-v8a)

```bash
# Set Android NDK path
export ANDROID_NDK=/path/to/android-ndk

# Build flashlight-text for Android
cd /tmp/text
mkdir build-android && cd build-android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -DFL_TEXT_USE_KENLM=ON \
    -DFL_TEXT_BUILD_STANDALONE=ON
make -j8
make install DESTDIR=/tmp/android-install

# Build KenLM for Android
cd /tmp/kenlm
mkdir build-android && cd build-android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release
make -j8

# Build Cued Speech Decoder for Android
cd /path/to/cued_speech/cpp
mkdir build-android && cd build-android
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_PREFIX_PATH=/tmp/android-install/usr/local
make -j8

# Output: libcued_speech_decoder.so
```

### For iOS (arm64)

```bash
# Build flashlight-text for iOS
cd /tmp/text
mkdir build-ios && cd build-ios
cmake .. \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DFL_TEXT_USE_KENLM=ON
make -j8
make install DESTDIR=/tmp/ios-install

# Build KenLM for iOS
cd /tmp/kenlm
mkdir build-ios && cd build-ios
cmake .. \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
    -DCMAKE_BUILD_TYPE=Release
make -j8

# Build Cued Speech Decoder for iOS
cd /path/to/cued_speech/cpp
mkdir build-ios && cd build-ios
cmake .. \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=12.0 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_PREFIX_PATH=/tmp/ios-install/usr/local
make -j8

# Output: libcued_speech_decoder.a
```

## Step 2: Add Libraries to Flutter Project

### Android

```bash
# Create jniLibs directory structure
cd your_flutter_app
mkdir -p android/app/src/main/jniLibs/arm64-v8a

# Copy libraries
cp /path/to/libcued_speech_decoder.so android/app/src/main/jniLibs/arm64-v8a/
cp /path/to/libflashlight-text.so android/app/src/main/jniLibs/arm64-v8a/
cp /path/to/libkenlm.so android/app/src/main/jniLibs/arm64-v8a/
```

Update `android/app/build.gradle`:

```gradle
android {
    ...
    sourceSets {
        main {
            jniLibs.srcDirs = ['src/main/jniLibs']
        }
    }
}
```

### iOS

```bash
# Create Frameworks directory
cd your_flutter_app
mkdir -p ios/Frameworks

# Copy static libraries
cp /path/to/libcued_speech_decoder.a ios/Frameworks/
cp /path/to/libflashlight-text.a ios/Frameworks/
cp /path/to/libkenlm.a ios/Frameworks/
```

Update `ios/Podspec` or manually in Xcode:

1. Open `ios/Runner.xcworkspace` in Xcode
2. Select Runner target → Build Phases → Link Binary With Libraries
3. Add the `.a` files
4. Add required system frameworks: `libc++.tbd`, `libz.tbd`, `libbz2.tbd`

## Step 3: Create Dart FFI Bindings

### Install FFI Package

Add to `pubspec.yaml`:

```yaml
dependencies:
  ffi: ^2.0.0
  
dev_dependencies:
  ffigen: ^8.0.0
```

### Generate Bindings Automatically (Recommended)

Create `ffigen.yaml`:

```yaml
name: CuedSpeechBindings
description: FFI bindings for Cued Speech Decoder
output: 'lib/src/bindings.dart'
headers:
  entry-points:
    - 'path/to/cpp/decoder_c_api.h'
```

Run:

```bash
dart run ffigen
```

### Or Write Bindings Manually

Create `lib/src/decoder_bindings.dart`:

```dart
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:ffi/ffi.dart';

// Load native library
final ffi.DynamicLibrary _nativeLib = () {
  if (Platform.isAndroid) {
    return ffi.DynamicLibrary.open('libcued_speech_decoder.so');
  } else if (Platform.isIOS) {
    return ffi.DynamicLibrary.process();
  }
  throw UnsupportedError('Platform not supported');
}();

// C Structs
class DecoderConfig extends ffi.Struct {
  external ffi.Pointer<Utf8> lexicon_path;
  external ffi.Pointer<Utf8> tokens_path;
  external ffi.Pointer<Utf8> lm_path;
  external ffi.Pointer<Utf8> lm_dict_path;
  
  @ffi.Int32()
  external int nbest;
  @ffi.Int32()
  external int beam_size;
  @ffi.Int32()
  external int beam_size_token;
  @ffi.Float()
  external double beam_threshold;
  @ffi.Float()
  external double lm_weight;
  @ffi.Float()
  external double word_score;
  @ffi.Float()
  external double unk_score;
  @ffi.Float()
  external double sil_score;
  @ffi.Bool()
  external bool log_add;
  
  external ffi.Pointer<Utf8> blank_token;
  external ffi.Pointer<Utf8> sil_token;
  external ffi.Pointer<Utf8> unk_word;
}

class RecognitionResult extends ffi.Struct {
  @ffi.Int32()
  external int frame_number;
  external ffi.Pointer<ffi.Pointer<Utf8>> phonemes;
  @ffi.Int32()
  external int phonemes_length;
  external ffi.Pointer<Utf8> french_sentence;
  @ffi.Float()
  external double confidence;
}

// Function typedefs
typedef DecoderCreateC = ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<DecoderConfig>);
typedef DecoderCreateDart = ffi.Pointer<ffi.Void> Function(
    ffi.Pointer<DecoderConfig>);

typedef DecoderDestroyC = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef DecoderDestroyDart = void Function(ffi.Pointer<ffi.Void>);

typedef StreamCreateC = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef StreamCreateDart = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);

typedef StreamPushFrameC = ffi.Bool Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef StreamPushFrameDart = bool Function(
    ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);

// Function lookups
final decoderCreate = _nativeLib
    .lookup<ffi.NativeFunction<DecoderCreateC>>('decoder_create')
    .asFunction<DecoderCreateDart>();

final decoderDestroy = _nativeLib
    .lookup<ffi.NativeFunction<DecoderDestroyC>>('decoder_destroy')
    .asFunction<DecoderDestroyDart>();

final streamCreate = _nativeLib
    .lookup<ffi.NativeFunction<StreamCreateC>>('stream_create')
    .asFunction<StreamCreateDart>();

final streamPushFrame = _nativeLib
    .lookup<ffi.NativeFunction<StreamPushFrameC>>('stream_push_frame')
    .asFunction<StreamPushFrameDart>();

final decoderGetVocabSize = _nativeLib
    .lookup<ffi.NativeFunction<ffi.Int32 Function(ffi.Pointer<ffi.Void>)>>(
        'decoder_get_vocab_size')
    .asFunction<int Function(ffi.Pointer<ffi.Void>)>();
```

## Step 4: Create Dart Wrapper Class

Create `lib/src/decoder.dart`:

```dart
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'decoder_bindings.dart' as bindings;

class CuedSpeechDecoder {
  ffi.Pointer<ffi.Void>? _decoderHandle;
  ffi.Pointer<ffi.Void>? _streamHandle;
  int? _vocabSize;
  
  /// Initialize decoder with model files
  Future<void> initialize({
    required String lexiconPath,
    required String tokensPath,
    required String lmPath,
    int beamSize = 40,
    double lmWeight = 3.23,
  }) async {
    final config = calloc<bindings.DecoderConfig>();
    
    config.ref.lexicon_path = lexiconPath.toNativeUtf8();
    config.ref.tokens_path = tokensPath.toNativeUtf8();
    config.ref.lm_path = lmPath.toNativeUtf8();
    config.ref.lm_dict_path = ffi.nullptr;
    config.ref.nbest = 1;
    config.ref.beam_size = beamSize;
    config.ref.beam_size_token = -1;
    config.ref.beam_threshold = 50.0;
    config.ref.lm_weight = lmWeight;
    config.ref.word_score = 0.0;
    config.ref.unk_score = double.negativeInfinity;
    config.ref.sil_score = 0.0;
    config.ref.log_add = false;
    config.ref.blank_token = '<BLANK>'.toNativeUtf8();
    config.ref.sil_token = '_'.toNativeUtf8();
    config.ref.unk_word = '<UNK>'.toNativeUtf8();
    
    _decoderHandle = bindings.decoderCreate(config);
    
    calloc.free(config);
    
    if (_decoderHandle == ffi.nullptr) {
      throw Exception('Failed to initialize decoder');
    }
    
    _vocabSize = bindings.decoderGetVocabSize(_decoderHandle!);
    _streamHandle = bindings.streamCreate(_decoderHandle!);
    
    if (_streamHandle == ffi.nullptr) {
      throw Exception('Failed to create stream');
    }
  }
  
  /// Push a frame of features [7 hand_shape + 18 hand_pos + 8 lips = 33]
  /// Returns true if a window is ready to process
  bool pushFrame(List<double> features) {
    if (_streamHandle == null) {
      throw StateError('Decoder not initialized');
    }
    
    if (features.length != 33) {
      throw ArgumentError('Features must have length 33');
    }
    
    final featuresPtr = calloc<ffi.Float>(33);
    for (int i = 0; i < 33; i++) {
      featuresPtr[i] = features[i];
    }
    
    final ready = bindings.streamPushFrame(_streamHandle!, featuresPtr);
    
    calloc.free(featuresPtr);
    return ready;
  }
  
  int? get vocabSize => _vocabSize;
  
  void dispose() {
    if (_streamHandle != null) {
      // Call stream_destroy if needed
      _streamHandle = null;
    }
    if (_decoderHandle != null) {
      bindings.decoderDestroy(_decoderHandle!);
      _decoderHandle = null;
    }
  }
}
```

## Step 5: Integrate with Your App

### Copy Model Files to Assets

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/lexicon.txt
    - assets/models/tokens.txt
    - assets/models/lm.bin
```

### Use in App

```dart
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'src/decoder.dart';

class DecoderService {
  final CuedSpeechDecoder decoder = CuedSpeechDecoder();
  
  Future<void> initialize() async {
    // Copy assets to local storage (for native file access)
    final appDir = await getApplicationDocumentsDirectory();
    final lexiconPath = '${appDir.path}/lexicon.txt';
    final tokensPath = '${appDir.path}/tokens.txt';
    final lmPath = '${appDir.path}/lm.bin';
    
    // Copy from assets if not exists
    await _copyAssetIfNeeded('assets/models/lexicon.txt', lexiconPath);
    await _copyAssetIfNeeded('assets/models/tokens.txt', tokensPath);
    await _copyAssetIfNeeded('assets/models/lm.bin', lmPath);
    
    // Initialize decoder
    await decoder.initialize(
      lexiconPath: lexiconPath,
      tokensPath: tokensPath,
      lmPath: lmPath,
      beamSize: 40,
      lmWeight: 3.23,
    );
  }
  
  Future<void> _copyAssetIfNeeded(String assetPath, String targetPath) async {
    final file = File(targetPath);
    if (!await file.exists()) {
      final data = await rootBundle.load(assetPath);
      await file.writeAsBytes(data.buffer.asUint8List());
    }
  }
  
  bool processFrame(List<double> features) {
    return decoder.pushFrame(features);
  }
  
  void dispose() {
    decoder.dispose();
  }
}
```

## Step 6: Handle TFLite Inference

Your Flutter app needs to run TFLite inference to get logits. Two approaches:

### Option A: TFLite in Dart (Simpler)

```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteRunner {
  Interpreter? _interpreter;
  
  Future<void> loadModel(String modelPath) async {
    _interpreter = await Interpreter.fromAsset(modelPath);
  }
  
  List<double> runInference(List<double> features, int windowSize) {
    // Reshape features [windowSize x 33] -> model input shape
    final input = Float32List.fromList(features);
    final output = List.filled(windowSize * vocabSize, 0.0);
    
    _interpreter!.run(input, output);
    return output;
  }
}
```

Then integrate with decoder's streaming:
- Push frames with `decoder.pushFrame(features)`
- When ready, run TFLite to get logits
- Pass logits to C++ via a separate FFI call (requires extending the C API)

### Option B: TFLite in C++ (Better Performance)

Move TFLite inference to C++ and simplify the callback. This requires:
1. Adding TFLite C++ to your native build
2. Loading the model in `decoder.cpp`
3. Running inference inside `WindowProcessor::process_window`

## Performance Tips

1. **Isolate for Background Processing**
   ```dart
   import 'dart:isolate';
   
   void decoderIsolate(SendPort sendPort) {
     // Run decoder in separate isolate
   }
   ```

2. **Memory-Mapped LM**
   - Ensure LM is memory-mapped (KenLM does this automatically)
   - Keep model files uncompressed in assets

3. **Reduce Copies**
   - Use `Pointer` directly when possible
   - Avoid repeated `toNativeUtf8()` conversions

4. **Profile**
   ```dart
   import 'dart:developer' as developer;
   
   developer.Timeline.startSync('decoder_push_frame');
   decoder.pushFrame(features);
   developer.Timeline.finishSync();
   ```

## Troubleshooting

### Library not found (Android)

Check logcat:
```bash
adb logcat | grep cued_speech
```

Verify library is included:
```bash
unzip -l app-release.apk | grep libcued_speech
```

### Symbol not found (iOS)

Check symbols:
```bash
nm -gU ios/Frameworks/libcued_speech_decoder.a | grep decoder_create
```

Ensure library is linked in Xcode build phases.

### Crash on first call

- Check that all model files exist and are accessible
- Verify file paths are correct (use absolute paths)
- Check error with `decoder_get_last_error()`

## Next Steps

1. Test on physical devices (both Android and iOS)
2. Profile and optimize hot paths
3. Add error handling and recovery
4. Implement UI for real-time display
5. Consider batching for better throughput

## References

- [Flutter FFI](https://dart.dev/guides/libraries/c-interop)
- [ffigen](https://pub.dev/packages/ffigen)
- [tflite_flutter](https://pub.dev/packages/tflite_flutter)

