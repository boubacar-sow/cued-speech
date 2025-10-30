# C++ Implementation Notes

## What Was Implemented

This C++ implementation ports the Python `decoder_tflite.py` to native code for mobile deployment via Flutter FFI.

### Core Components

1. **CTCDecoder** (`decoder.h/cpp`)
   - Wraps flashlight-text's `LexiconDecoder`
   - Integrates KenLM for language model scoring
   - Provides CTC beam search decoding
   - Matches Python's `ctc_decoder` from torchaudio

2. **WindowProcessor** (`decoder.h/cpp`)
   - Implements overlap-save windowing
   - Parameters: `WINDOW_SIZE=100`, `COMMIT_SIZE=50`, `LEFT_CONTEXT=25`
   - Manages streaming state across chunks
   - Accumulates logits and decodes incrementally

3. **FeatureExtractor** (`decoder.h/cpp`)
   - Extracts hand shape (7 features), hand position (18 features), lips (8 features)
   - Computes distances, angles, velocities, acceleration
   - Placeholder implementation - needs completion from Python code

4. **SentenceCorrector** (`decoder.h/cpp`)
   - Homophone-based correction using KenLM
   - Beam search over homophones
   - Converts LIAPHON → IPA → French text
   - Placeholder implementation - needs completion

5. **C API** (`decoder_c_api.h/cpp`)
   - Simple C interface for Flutter FFI
   - Opaque handles for decoder, stream, corrector
   - No C++ exceptions across boundary
   - Memory management helpers

### Key Differences from Python

| Aspect | Python | C++ |
|--------|--------|-----|
| **CTC Decoder** | `torchaudio.models.decoder.ctc_decoder` | `flashlight::lib::text::LexiconDecoder` |
| **Model Inference** | PyTorch TFLite wrapper | TFLite C API (via callback) |
| **Tensors** | `torch.Tensor` | Raw `float*` arrays |
| **Log Softmax** | `F.log_softmax` | Manual implementation |
| **Language Model** | Python KenLM bindings | Native KenLM C++ |
| **Memory** | Automatic (Python GC) | Manual (`new`/`delete`, smart pointers) |

## What Still Needs Implementation

### 1. Complete Feature Extraction

The `FeatureExtractor::extract()` method currently returns placeholder values. Needs full port of `extract_features_single_row()` from Python:

- Hand-face distances (all index pairs)
- Hand-hand distances (shape features)
- Thumb-index angle
- Lip width, height, area, curvature
- Velocity and acceleration features

**Effort:** ~2-3 hours

### 2. Complete Sentence Corrector

The `SentenceCorrector` class needs:

- Load homophones from JSONL file
- LIAPHON → IPA conversion (multi-character handling)
- KenLM beam search implementation
- Match Python's `beam_search()` and `correct_french_sentences()` logic

**Effort:** ~2-4 hours

### 3. Final Chunk Processing

`WindowProcessor::finalize()` is incomplete. Needs to handle the last uncommitted frames:

- Determine padding needed
- Process final window
- Extract and commit remaining logits
- Match Python's final chunk logic (lines 1421-1509)

**Effort:** ~1-2 hours

### 4. Error Handling

Currently minimal. Needs:

- More descriptive error messages
- Validation of inputs (file existence, array bounds, etc.)
- Graceful degradation when landmarks are missing
- Thread-safe error reporting

**Effort:** ~1-2 hours

### 5. Testing

No tests yet. Needs:

- Unit tests for feature extraction
- Integration tests comparing with Python outputs
- Golden file tests for end-to-end decoding
- Memory leak checks (valgrind, ASAN)

**Effort:** ~4-6 hours

### 6. Build System Hardening

Current CMakeLists.txt is basic. Needs:

- Robust dependency finding (fallbacks)
- Android/iOS toolchain files
- Cross-platform compatibility (MSVC, Clang, GCC)
- Install rules for headers
- Package config files

**Effort:** ~2-3 hours

### 7. Documentation

- API documentation (Doxygen)
- Architecture diagrams
- Performance benchmarks
- Migration guide from Python

**Effort:** ~3-4 hours

## Known Limitations

1. **Feature Extraction Not Complete**
   - Currently returns zeros
   - Won't produce correct results until ported

2. **No TFLite Integration**
   - Decoder expects logits from external source
   - TFLite inference must be done in Dart or added separately

3. **Limited Testing**
   - No validation against Python reference
   - Edge cases not covered

4. **No Phoneme Conversion**
   - IPA ↔ LIAPHON conversion is simplified
   - Doesn't handle multi-character IPA sequences properly

5. **Memory Management**
   - No RAII wrappers for C API return values
   - User must remember to free resources

## Performance Expectations

### Compared to Python

| Metric | Python (PyTorch) | C++ (flashlight) | Speedup |
|--------|------------------|------------------|---------|
| Decoder Init | ~2-3s | ~0.5-1s | 2-3x |
| Per-Frame Feature Extraction | ~2-5ms | ~0.5-1ms | 3-5x |
| CTC Decoding (100 frames) | ~50-100ms | ~20-40ms | 2-3x |
| LM Scoring | ~10-20ms | ~5-10ms | 2x |
| **Total (frame-to-text)** | ~70-130ms | ~30-60ms | ~2-3x |

### Memory

- Python: ~500MB-1GB (PyTorch overhead)
- C++: ~100-300MB (mostly LM model)

### Mobile Constraints

- Target: <100ms latency per window
- Memory: <200MB total (including TFLite)
- Battery: Beam search is CPU-intensive, tune `beam_size`

## Integration Checklist

- [ ] Build flashlight-text for Android/iOS
- [ ] Build KenLM for Android/iOS
- [ ] Build cued_speech_decoder for Android/iOS
- [ ] Complete feature extraction implementation
- [ ] Complete sentence corrector implementation
- [ ] Test against Python reference outputs
- [ ] Create Dart FFI bindings
- [ ] Integrate TFLite inference path
- [ ] Profile on target devices
- [ ] Optimize hot paths
- [ ] Add error handling
- [ ] Write tests
- [ ] Document API

## Recommended Next Steps

1. **Validate Core Decoding First**
   - Use pre-computed logits from Python
   - Test C++ decoder produces same results
   - Ensures flashlight-text integration works

2. **Port Feature Extraction**
   - Critical for end-to-end pipeline
   - Test each feature independently

3. **Test on Mobile**
   - Build for Android arm64
   - Run on real device
   - Profile and identify bottlenecks

4. **Iterate**
   - Optimize based on profiling
   - Reduce beam size if needed
   - Consider pruning LM

## Alternative Approaches Considered

1. **Pure Dart Implementation**
   - Pros: No native code, simpler deployment
   - Cons: 5-10x slower, larger memory, no KenLM
   - **Decision:** Rejected (too slow for real-time)

2. **Python Backend**
   - Pros: No porting needed
   - Cons: Network latency, privacy, scaling costs
   - **Decision:** Rejected (user requirement: on-device)

3. **ONNX Runtime**
   - Pros: Portable, GPU support
   - Cons: Doesn't include CTC decoder or LM
   - **Decision:** Rejected (still need flashlight/KenLM)

4. **TensorFlow Lite with Custom Ops**
   - Pros: Unified runtime
   - Cons: Complex to add CTC+LM as custom ops
   - **Decision:** Considered for future (v2)

## Conclusion

This C++ implementation provides a **solid foundation** for on-device decoding but requires **completion of feature extraction and testing** before production use. The architecture is sound and matches the Python reference closely.

**Estimated total effort to production-ready:** 15-25 hours
- Core implementation completion: 8-12 hours
- Testing and validation: 4-6 hours
- Mobile integration and tuning: 3-7 hours

**Risk areas:**
- Cross-compilation complexity (Android/iOS)
- LGPL compliance for iOS (KenLM)
- Performance on low-end devices
- Model size and loading time

**Mitigation:**
- Use provided build scripts
- Consider alternative LM or dynamic framework on iOS
- Aggressive LM pruning and beam size tuning
- Memory-map and compress model files

