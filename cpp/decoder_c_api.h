/**
 * C API for Cued Speech Decoder
 * 
 * This provides a simple C interface that can be called from Flutter via FFI.
 */

#ifndef CUED_SPEECH_DECODER_C_API_H
#define CUED_SPEECH_DECODER_C_API_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef void* DecoderHandle;
typedef void* StreamHandle;

/**
 * Configuration for decoder initialization
 */
typedef struct {
    const char* lexicon_path;
    const char* tokens_path;
    const char* lm_path;
    const char* lm_dict_path;  // Can be NULL
    
    int nbest;
    int beam_size;
    int beam_size_token;
    float beam_threshold;
    float lm_weight;
    float word_score;
    float unk_score;
    float sil_score;
    bool log_add;
    
    const char* blank_token;
    const char* sil_token;
    const char* unk_word;
} DecoderConfig;

/**
 * Default decoder configuration
 */
DecoderConfig decoder_config_default();

/**
 * Hypothesis result
 */
typedef struct {
    int* tokens;              // Token indices
    int tokens_length;
    char** words;             // Decoded words (NULL-terminated strings)
    int words_length;
    float score;
    int* timesteps;           // Token timesteps
    int timesteps_length;
} Hypothesis;

/**
 * Recognition result
 */
typedef struct {
    int frame_number;
    char** phonemes;          // NULL-terminated strings
    int phonemes_length;
    char* french_sentence;    // NULL-terminated string (can be NULL)
    float confidence;
} RecognitionResult;

//=============================================================================
// Decoder Lifecycle
//=============================================================================

/**
 * Create and initialize a decoder
 * 
 * @param config Decoder configuration
 * @return Decoder handle, or NULL on failure
 */
DecoderHandle decoder_create(const DecoderConfig* config);

/**
 * Destroy a decoder and free resources
 * 
 * @param handle Decoder handle
 */
void decoder_destroy(DecoderHandle handle);

/**
 * Get vocabulary size
 * 
 * @param handle Decoder handle
 * @return Vocabulary size
 */
int decoder_get_vocab_size(DecoderHandle handle);

/**
 * Convert token index to string
 * 
 * @param handle Decoder handle
 * @param idx Token index
 * @return Token string (caller must NOT free), or NULL if invalid
 */
const char* decoder_idx_to_token(DecoderHandle handle, int idx);

/**
 * Convert token string to index
 * 
 * @param handle Decoder handle
 * @param token Token string
 * @return Token index, or -1 if not found
 */
int decoder_token_to_idx(DecoderHandle handle, const char* token);

//=============================================================================
// Single-Shot Decoding
//=============================================================================

/**
 * Decode a complete sequence of logits
 * 
 * @param handle Decoder handle
 * @param logits Logits array [T x V] in row-major order
 * @param T Number of time steps
 * @param V Vocabulary size
 * @param[out] num_results Number of hypotheses returned
 * @return Array of hypotheses (caller must free with decoder_free_hypotheses)
 */
Hypothesis* decoder_decode(
    DecoderHandle handle,
    const float* logits,
    int T,
    int V,
    int* num_results
);

/**
 * Decode from log probabilities
 * 
 * @param handle Decoder handle
 * @param log_probs Log probabilities [T x V] in row-major order
 * @param T Number of time steps
 * @param V Vocabulary size
 * @param[out] num_results Number of hypotheses returned
 * @return Array of hypotheses (caller must free with decoder_free_hypotheses)
 */
Hypothesis* decoder_decode_log_probs(
    DecoderHandle handle,
    const float* log_probs,
    int T,
    int V,
    int* num_results
);

/**
 * Free hypotheses returned by decoder_decode
 * 
 * @param hypotheses Hypothesis array
 * @param num_results Number of hypotheses
 */
void decoder_free_hypotheses(Hypothesis* hypotheses, int num_results);

//=============================================================================
// Streaming Decoding with Windowing
//=============================================================================

/**
 * Create a new streaming session
 * 
 * @param decoder_handle Decoder handle
 * @return Stream handle, or NULL on failure
 */
StreamHandle stream_create(DecoderHandle decoder_handle);

/**
 * Load a TFLite sequence model for the stream.
 *
 * @param handle Stream handle
 * @param model_path Path to the TFLite model file
 * @return true on success, false otherwise
 */
bool stream_load_tflite_model(StreamHandle handle, const char* model_path);

/**
 * Destroy a streaming session
 * 
 * @param handle Stream handle
 */
void stream_destroy(StreamHandle handle);

/**
 * Reset a stream to start processing a new sequence
 * 
 * @param handle Stream handle
 */
void stream_reset(StreamHandle handle);

/**
 * Push a frame of features to the stream
 * 
 * The features should be organized as:
 * - hand_shape: 7 floats
 * - hand_position: 18 floats
 * - lips: 8 floats
 * Total: 33 floats
 * 
 * @param handle Stream handle
 * @param features Feature array [33 floats]
 * @return true if window is ready to process, false otherwise
 */
bool stream_push_frame(StreamHandle handle, const float* features);

/**
 * Process current window and get partial result
 *
 * This should be called when stream_push_frame returns true.
 *
 * @param handle Stream handle
 * @return Recognition result (caller must free with stream_free_result)
 */
RecognitionResult* stream_process_window(StreamHandle handle);

/**
 * Finalize stream and get final result
 * 
 * @param handle Stream handle
 * @return Recognition result (caller must free with stream_free_result)
 */
RecognitionResult* stream_finalize(StreamHandle handle);

/**
 * Free recognition result
 * 
 * @param result Recognition result
 */
void stream_free_result(RecognitionResult* result);

//=============================================================================
// Sentence Correction
//=============================================================================

/**
 * Opaque handle for sentence corrector
 */
typedef void* CorrectorHandle;

/**
 * Create a sentence corrector
 * 
 * @param homophones_path Path to homophones JSONL file
 * @param kenlm_path Path to KenLM model for French
 * @return Corrector handle, or NULL on failure
 */
CorrectorHandle corrector_create(
    const char* homophones_path,
    const char* kenlm_path
);

/**
 * Destroy a sentence corrector
 * 
 * @param handle Corrector handle
 */
void corrector_destroy(CorrectorHandle handle);

/**
 * Correct LIAPHON phoneme sequence to French text
 * 
 * @param handle Corrector handle
 * @param phonemes Array of LIAPHON phoneme strings
 * @param num_phonemes Number of phonemes
 * @return French sentence (caller must free with corrector_free_string)
 */
char* corrector_correct(
    CorrectorHandle handle,
    const char** phonemes,
    int num_phonemes
);

/**
 * Free string returned by corrector_correct
 * 
 * @param str String to free
 */
void corrector_free_string(char* str);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * Get last error message
 * 
 * @return Error message (do not free)
 */
const char* decoder_get_last_error();

/**
 * Convert LIAPHON phonemes to IPA string
 * 
 * @param phonemes Array of LIAPHON phoneme strings
 * @param num_phonemes Number of phonemes
 * @return IPA string (caller must free)
 */
char* phoneme_liaphon_to_ipa(const char** phonemes, int num_phonemes);

/**
 * Convert IPA string to LIAPHON phonemes
 * 
 * @param ipa IPA string
 * @param[out] num_phonemes Number of phonemes returned
 * @return Array of LIAPHON phoneme strings (caller must free)
 */
char** phoneme_ipa_to_liaphon(const char* ipa, int* num_phonemes);

/**
 * Free string array
 * 
 * @param strings String array
 * @param count Number of strings
 */
void decoder_free_string_array(char** strings, int count);

#ifdef __cplusplus
}
#endif

#endif // CUED_SPEECH_DECODER_C_API_H

