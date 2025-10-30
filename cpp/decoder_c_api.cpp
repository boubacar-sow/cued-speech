/**
 * C API for Cued Speech Decoder - Implementation
 */

#include "decoder_c_api.h"
#include "decoder.h"

#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <iostream>

using namespace cued_speech;

// Thread-local error message
thread_local std::string g_last_error;

void set_last_error(const std::string& error) {
    g_last_error = error;
    std::cerr << "Error: " << error << std::endl;
}

//=============================================================================
// Helper Functions
//=============================================================================

char* copy_string(const std::string& str) {
    char* result = new char[str.length() + 1];
    std::strcpy(result, str.c_str());
    return result;
}

char** copy_string_vector(const std::vector<std::string>& vec) {
    if (vec.empty()) {
        return nullptr;
    }
    
    char** result = new char*[vec.size()];
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = copy_string(vec[i]);
    }
    return result;
}

//=============================================================================
// Decoder Configuration
//=============================================================================

DecoderConfig decoder_config_default() {
    DecoderConfig config;
    config.lexicon_path = nullptr;
    config.tokens_path = nullptr;
    config.lm_path = nullptr;
    config.lm_dict_path = nullptr;
    config.nbest = 1;
    config.beam_size = 40;
    config.beam_size_token = -1;
    config.beam_threshold = 50.0f;
    config.lm_weight = 3.23f;
    config.word_score = 0.0f;
    config.unk_score = -std::numeric_limits<float>::infinity();
    config.sil_score = 0.0f;
    config.log_add = false;
    config.blank_token = "<BLANK>";
    config.sil_token = "_";
    config.unk_word = "<UNK>";
    return config;
}

//=============================================================================
// Decoder Lifecycle
//=============================================================================

DecoderHandle decoder_create(const DecoderConfig* config) {
    try {
        if (!config) {
            set_last_error("Config is NULL");
            return nullptr;
        }
        
        DecoderConfig cpp_config;
        cpp_config.lexicon_path = config->lexicon_path ? config->lexicon_path : "";
        cpp_config.tokens_path = config->tokens_path ? config->tokens_path : "";
        cpp_config.lm_path = config->lm_path ? config->lm_path : "";
        cpp_config.lm_dict_path = config->lm_dict_path ? config->lm_dict_path : "";
        cpp_config.nbest = config->nbest;
        cpp_config.beam_size = config->beam_size;
        cpp_config.beam_size_token = config->beam_size_token;
        cpp_config.beam_threshold = config->beam_threshold;
        cpp_config.lm_weight = config->lm_weight;
        cpp_config.word_score = config->word_score;
        cpp_config.unk_score = config->unk_score;
        cpp_config.sil_score = config->sil_score;
        cpp_config.log_add = config->log_add;
        cpp_config.blank_token = config->blank_token;
        cpp_config.sil_token = config->sil_token;
        cpp_config.unk_word = config->unk_word;
        
        auto decoder = std::make_unique<CTCDecoder>(cpp_config);
        
        if (!decoder->initialize()) {
            set_last_error("Failed to initialize decoder");
            return nullptr;
        }
        
        return decoder.release();
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in decoder_create: ") + e.what());
        return nullptr;
    }
}

void decoder_destroy(DecoderHandle handle) {
    if (handle) {
        delete static_cast<CTCDecoder*>(handle);
    }
}

int decoder_get_vocab_size(DecoderHandle handle) {
    if (!handle) {
        return 0;
    }
    
    try {
        return static_cast<CTCDecoder*>(handle)->get_vocab_size();
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in decoder_get_vocab_size: ") + e.what());
        return 0;
    }
}

const char* decoder_idx_to_token(DecoderHandle handle, int idx) {
    if (!handle) {
        return nullptr;
    }
    
    try {
        auto decoder = static_cast<CTCDecoder*>(handle);
        static thread_local std::string token_buffer;
        token_buffer = decoder->idx_to_token(idx);
        return token_buffer.c_str();
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in decoder_idx_to_token: ") + e.what());
        return nullptr;
    }
}

int decoder_token_to_idx(DecoderHandle handle, const char* token) {
    if (!handle || !token) {
        return -1;
    }
    
    try {
        return static_cast<CTCDecoder*>(handle)->token_to_idx(token);
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in decoder_token_to_idx: ") + e.what());
        return -1;
    }
}

//=============================================================================
// Single-Shot Decoding
//=============================================================================

Hypothesis* decoder_decode(
    DecoderHandle handle,
    const float* logits,
    int T,
    int V,
    int* num_results) {
    
    if (!handle || !logits || !num_results) {
        set_last_error("Invalid arguments to decoder_decode");
        *num_results = 0;
        return nullptr;
    }
    
    try {
        auto decoder = static_cast<CTCDecoder*>(handle);
        auto results = decoder->decode(logits, T, V);
        
        *num_results = results.size();
        
        if (results.empty()) {
            return nullptr;
        }
        
        Hypothesis* hypotheses = new Hypothesis[results.size()];
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            
            // Copy tokens
            hypotheses[i].tokens_length = result.tokens.size();
            hypotheses[i].tokens = new int[result.tokens.size()];
            std::copy(result.tokens.begin(), result.tokens.end(), hypotheses[i].tokens);
            
            // Copy words
            hypotheses[i].words_length = result.words.size();
            hypotheses[i].words = copy_string_vector(result.words);
            
            // Copy score
            hypotheses[i].score = result.score;
            
            // Copy timesteps
            hypotheses[i].timesteps_length = result.timesteps.size();
            hypotheses[i].timesteps = new int[result.timesteps.size()];
            std::copy(result.timesteps.begin(), result.timesteps.end(), 
                     hypotheses[i].timesteps);
        }
        
        return hypotheses;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in decoder_decode: ") + e.what());
        *num_results = 0;
        return nullptr;
    }
}

Hypothesis* decoder_decode_log_probs(
    DecoderHandle handle,
    const float* log_probs,
    int T,
    int V,
    int* num_results) {
    
    if (!handle || !log_probs || !num_results) {
        set_last_error("Invalid arguments to decoder_decode_log_probs");
        *num_results = 0;
        return nullptr;
    }
    
    try {
        auto decoder = static_cast<CTCDecoder*>(handle);
        auto results = decoder->decode_log_probs(log_probs, T, V);
        
        *num_results = results.size();
        
        if (results.empty()) {
            return nullptr;
        }
        
        Hypothesis* hypotheses = new Hypothesis[results.size()];
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            
            hypotheses[i].tokens_length = result.tokens.size();
            hypotheses[i].tokens = new int[result.tokens.size()];
            std::copy(result.tokens.begin(), result.tokens.end(), hypotheses[i].tokens);
            
            hypotheses[i].words_length = result.words.size();
            hypotheses[i].words = copy_string_vector(result.words);
            
            hypotheses[i].score = result.score;
            
            hypotheses[i].timesteps_length = result.timesteps.size();
            hypotheses[i].timesteps = new int[result.timesteps.size()];
            std::copy(result.timesteps.begin(), result.timesteps.end(), 
                     hypotheses[i].timesteps);
        }
        
        return hypotheses;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in decoder_decode_log_probs: ") + e.what());
        *num_results = 0;
        return nullptr;
    }
}

void decoder_free_hypotheses(Hypothesis* hypotheses, int num_results) {
    if (!hypotheses) {
        return;
    }
    
    for (int i = 0; i < num_results; ++i) {
        delete[] hypotheses[i].tokens;
        
        if (hypotheses[i].words) {
            for (int j = 0; j < hypotheses[i].words_length; ++j) {
                delete[] hypotheses[i].words[j];
            }
            delete[] hypotheses[i].words;
        }
        
        delete[] hypotheses[i].timesteps;
    }
    
    delete[] hypotheses;
}

//=============================================================================
// Streaming Decoding
//=============================================================================

struct StreamContext {
    CTCDecoder* decoder;
    std::unique_ptr<TFLiteSequenceModel> sequence_model;
    std::unique_ptr<WindowProcessor> processor;
};

StreamHandle stream_create(DecoderHandle decoder_handle) {
    if (!decoder_handle) {
        set_last_error("Invalid decoder handle");
        return nullptr;
    }
    
    try {
        auto decoder = static_cast<CTCDecoder*>(decoder_handle);

        auto ctx = new StreamContext;
        ctx->decoder = decoder;
        ctx->sequence_model = std::make_unique<TFLiteSequenceModel>();
        ctx->processor = std::make_unique<WindowProcessor>(decoder, ctx->sequence_model.get());
        
        return ctx;
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in stream_create: ") + e.what());
        return nullptr;
    }
}

bool stream_load_tflite_model(StreamHandle handle, const char* model_path) {
    if (!handle || !model_path) {
        set_last_error("Invalid arguments to stream_load_tflite_model");
        return false;
    }

    try {
        auto ctx = static_cast<StreamContext*>(handle);
        if (!ctx->sequence_model) {
            ctx->sequence_model = std::make_unique<TFLiteSequenceModel>();
            ctx->processor = std::make_unique<WindowProcessor>(ctx->decoder, ctx->sequence_model.get());
        }
        if (!ctx->sequence_model->load(model_path)) {
            set_last_error("Failed to load TFLite sequence model");
            return false;
        }
        ctx->processor->reset();
        return true;
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in stream_load_tflite_model: ") + e.what());
        return false;
    }
}

void stream_destroy(StreamHandle handle) {
    if (handle) {
        auto ctx = static_cast<StreamContext*>(handle);
        delete ctx;
    }
}

void stream_reset(StreamHandle handle) {
    if (!handle) {
        return;
    }
    
    try {
        auto ctx = static_cast<StreamContext*>(handle);
        ctx->processor->reset();
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in stream_reset: ") + e.what());
    }
}

bool stream_push_frame(StreamHandle handle, const float* features) {
    if (!handle || !features) {
        set_last_error("Invalid arguments to stream_push_frame");
        return false;
    }
    
    try {
        auto ctx = static_cast<StreamContext*>(handle);
        
        FrameFeatures frame;
        frame.hand_shape.assign(features, features + 7);
        frame.hand_position.assign(features + 7, features + 25);
        frame.lips.assign(features + 25, features + 33);
        
        return ctx->processor->push_frame(frame);
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in stream_push_frame: ") + e.what());
        return false;
    }
}

RecognitionResult* stream_process_window(StreamHandle handle) {
    if (!handle) {
        set_last_error("Invalid stream handle");
        return nullptr;
    }

    try {
        auto ctx = static_cast<StreamContext*>(handle);
        auto result = ctx->processor->process_window();

        auto c_result = new RecognitionResult;
        c_result->frame_number = result.frame_number;
        c_result->phonemes_length = result.phonemes.size();
        c_result->phonemes = copy_string_vector(result.phonemes);
        c_result->french_sentence = result.french_sentence.empty()
            ? nullptr
            : copy_string(result.french_sentence);
        c_result->confidence = result.confidence;

        return c_result;
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in stream_process_window: ") + e.what());
        return nullptr;
    }
}

RecognitionResult* stream_finalize(StreamHandle handle) {
    if (!handle) {
        set_last_error("Invalid stream handle");
        return nullptr;
    }

    try {
        auto ctx = static_cast<StreamContext*>(handle);
        auto result = ctx->processor->finalize();

        auto c_result = new RecognitionResult;
        c_result->frame_number = result.frame_number;
        c_result->phonemes_length = result.phonemes.size();
        c_result->phonemes = copy_string_vector(result.phonemes);
        c_result->french_sentence = result.french_sentence.empty()
            ? nullptr
            : copy_string(result.french_sentence);
        c_result->confidence = result.confidence;

        return c_result;
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in stream_finalize: ") + e.what());
        return nullptr;
    }
}

void stream_free_result(RecognitionResult* result) {
    if (!result) {
        return;
    }
    
    if (result->phonemes) {
        for (int i = 0; i < result->phonemes_length; ++i) {
            delete[] result->phonemes[i];
        }
        delete[] result->phonemes;
    }
    
    if (result->french_sentence) {
        delete[] result->french_sentence;
    }
    
    delete result;
}

//=============================================================================
// Sentence Correction
//=============================================================================

CorrectorHandle corrector_create(
    const char* homophones_path,
    const char* kenlm_path) {
    
    if (!homophones_path || !kenlm_path) {
        set_last_error("Invalid paths to corrector_create");
        return nullptr;
    }
    
    try {
        auto corrector = std::make_unique<SentenceCorrector>(
            homophones_path, kenlm_path
        );
        
        if (!corrector->initialize()) {
            set_last_error("Failed to initialize sentence corrector");
            return nullptr;
        }
        
        return corrector.release();
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in corrector_create: ") + e.what());
        return nullptr;
    }
}

void corrector_destroy(CorrectorHandle handle) {
    if (handle) {
        delete static_cast<SentenceCorrector*>(handle);
    }
}

char* corrector_correct(
    CorrectorHandle handle,
    const char** phonemes,
    int num_phonemes) {
    
    if (!handle || !phonemes) {
        set_last_error("Invalid arguments to corrector_correct");
        return nullptr;
    }
    
    try {
        auto corrector = static_cast<SentenceCorrector*>(handle);
        
        std::vector<std::string> phoneme_vec;
        for (int i = 0; i < num_phonemes; ++i) {
            phoneme_vec.push_back(phonemes[i]);
        }
        
        std::string result = corrector->correct(phoneme_vec);
        return copy_string(result);
        
    } catch (const std::exception& e) {
        set_last_error(std::string("Exception in corrector_correct: ") + e.what());
        return nullptr;
    }
}

void corrector_free_string(char* str) {
    delete[] str;
}

//=============================================================================
// Utility Functions
//=============================================================================

const char* decoder_get_last_error() {
    return g_last_error.c_str();
}

char* phoneme_liaphon_to_ipa(const char** phonemes, int num_phonemes) {
    std::vector<std::string> phoneme_vec;
    for (int i = 0; i < num_phonemes; ++i) {
        phoneme_vec.push_back(phonemes[i]);
    }
    
    std::string ipa = liaphon_to_ipa(phoneme_vec);
    return copy_string(ipa);
}

char** phoneme_ipa_to_liaphon(const char* ipa, int* num_phonemes) {
    if (!ipa || !num_phonemes) {
        *num_phonemes = 0;
        return nullptr;
    }
    
    auto liaphon = ipa_to_liaphon(ipa);
    *num_phonemes = liaphon.size();
    
    return copy_string_vector(liaphon);
}

void decoder_free_string_array(char** strings, int count) {
    if (!strings) {
        return;
    }
    
    for (int i = 0; i < count; ++i) {
        delete[] strings[i];
    }
    delete[] strings;
}

