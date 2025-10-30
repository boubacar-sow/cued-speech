/**
 * Cued Speech Decoder - C++ Implementation
 */

#include "decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <limits>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <unordered_map>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <kenlm/lm/model.hh>
#include <opencv2/opencv.hpp>

// Flashlight-text includes
#include <flashlight/lib/text/decoder/LexiconDecoder.h>
#include <flashlight/lib/text/decoder/LexiconFreeDecoder.h>
#include <flashlight/lib/text/decoder/Trie.h>
#include <flashlight/lib/text/decoder/lm/KenLM.h>
#include <flashlight/lib/text/dictionary/Dictionary.h>
#include <flashlight/lib/text/dictionary/Utils.h>

namespace cued_speech {

struct TFLiteSequenceModel::Impl {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::array<int, 3> input_indices{};
    int output_index = -1;
    int vocab_size = 0;
    int last_sequence_length = 0;
    bool needs_allocation = true;
    bool loaded = false;
    std::mutex mutex;

    bool load(const std::string& model_path) {
        std::lock_guard<std::mutex> lock(mutex);

        model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!model) {
            return false;
        }

        tflite::InterpreterBuilder builder(*model, resolver);
        builder(&interpreter);
        if (!interpreter) {
            model.reset();
            return false;
        }

        if (interpreter->inputs().size() != 3) {
            throw std::runtime_error("TFLite model must have exactly 3 inputs (lips, hand_shape, hand_pos)");
        }

        for (int i = 0; i < 3; ++i) {
            input_indices[i] = interpreter->inputs()[i];
        }

        if (interpreter->outputs().empty()) {
            throw std::runtime_error("TFLite model must have at least one output");
        }

        output_index = interpreter->outputs()[0];
        needs_allocation = true;
        vocab_size = 0;
        last_sequence_length = 0;
        loaded = true;
        return true;
    }

    std::vector<float> infer(const std::vector<FrameFeatures>& frames, int window_size) {
        std::lock_guard<std::mutex> lock(mutex);

        if (!loaded || !interpreter) {
            return {};
        }

        const int seq_len = window_size > 0 ? window_size : static_cast<int>(frames.size());
        if (seq_len <= 0) {
            return {};
        }

        constexpr int kLipsDim = 8;
        constexpr int kHandShapeDim = 7;
        constexpr int kHandPosDim = 18;

        auto ensure_resize = [&](int input_idx, int dim) {
            TfLiteTensor* tensor = interpreter->tensor(input_idx);
            if (!tensor || !tensor->dims || tensor->dims->size != 3 || tensor->dims->data[1] != seq_len) {
                interpreter->ResizeInputTensor(input_idx, {1, seq_len, dim});
                needs_allocation = true;
            }
        };

        ensure_resize(input_indices[0], kLipsDim);
        ensure_resize(input_indices[1], kHandShapeDim);
        ensure_resize(input_indices[2], kHandPosDim);

        if (needs_allocation) {
            if (interpreter->AllocateTensors() != kTfLiteOk) {
                throw std::runtime_error("Failed to allocate TFLite tensors");
            }
            needs_allocation = false;
        }

        auto fill_input = [](float* dest, int dim, const std::vector<float>& source, int t) {
            for (int d = 0; d < dim; ++d) {
                float value = (d < static_cast<int>(source.size())) ? source[d] : 0.0f;
                dest[t * dim + d] = value;
            }
        };

        float* lips_input = interpreter->typed_input_tensor<float>(input_indices[0]);
        float* hand_shape_input = interpreter->typed_input_tensor<float>(input_indices[1]);
        float* hand_pos_input = interpreter->typed_input_tensor<float>(input_indices[2]);

        const FrameFeatures zero_frame{
            std::vector<float>(kHandShapeDim, 0.0f),
            std::vector<float>(kHandPosDim, 0.0f),
            std::vector<float>(kLipsDim, 0.0f)
        };

        for (int t = 0; t < seq_len; ++t) {
            const FrameFeatures& frame = (t < static_cast<int>(frames.size())) ? frames[t] : zero_frame;

            fill_input(lips_input, kLipsDim, frame.lips, t);
            fill_input(hand_shape_input, kHandShapeDim, frame.hand_shape, t);
            fill_input(hand_pos_input, kHandPosDim, frame.hand_position, t);
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Failed to invoke TFLite model");
        }

        TfLiteTensor* output = interpreter->tensor(output_index);
        if (!output || !output->dims || output->dims->size < 3) {
            throw std::runtime_error("Unexpected TFLite output tensor shape");
        }

        last_sequence_length = output->dims->data[output->dims->size - 2];
        vocab_size = output->dims->data[output->dims->size - 1];

        if (last_sequence_length <= 0 || vocab_size <= 0) {
            return {};
        }

        const float* output_data = interpreter->typed_output_tensor<float>(0);
        return std::vector<float>(
            output_data,
            output_data + static_cast<size_t>(last_sequence_length) * vocab_size
        );
    }

    bool is_loaded() const {
        return loaded && interpreter != nullptr;
    }
};

TFLiteSequenceModel::TFLiteSequenceModel()
    : impl_(std::make_unique<Impl>()) {}

TFLiteSequenceModel::~TFLiteSequenceModel() = default;

bool TFLiteSequenceModel::load(const std::string& model_path) {
    return impl_ ? impl_->load(model_path) : false;
}

std::vector<float> TFLiteSequenceModel::infer(const std::vector<FrameFeatures>& frames, int window_size) {
    return impl_ ? impl_->infer(frames, window_size) : std::vector<float>{};
}

int TFLiteSequenceModel::vocab_size() const {
    return impl_ ? impl_->vocab_size : 0;
}

int TFLiteSequenceModel::last_sequence_length() const {
    return impl_ ? impl_->last_sequence_length : 0;
}

bool TFLiteSequenceModel::is_loaded() const {
    return impl_ ? impl_->is_loaded() : false;
}

// Phoneme mappings
const std::map<std::string, std::string> IPA_TO_LIAPHON = {
    {"a", "a"}, {"ə", "x"}, {"ɛ", "e^"}, {"œ", "x^"},
    {"i", "i"}, {"y", "y"}, {"e", "e"}, {"u", "u"},
    {"ɔ", "o"}, {"o", "o^"}, {"ɑ̃", "a~"}, {"ɛ̃", "e~"},
    {"ɔ̃", "o~"}, {"œ̃", "x~"}, {" ", "_"}, {"b", "b"},
    {"c", "k"}, {"d", "d"}, {"f", "f"}, {"ɡ", "g"},
    {"j", "j"}, {"k", "k"}, {"l", "l"}, {"m", "m"},
    {"n", "n"}, {"p", "p"}, {"s", "s"}, {"t", "t"},
    {"v", "v"}, {"w", "w"}, {"z", "z"}, {"ɥ", "h"},
    {"ʁ", "r"}, {"ʃ", "s^"}, {"ʒ", "z^"}, {"ɲ", "gn"},
    {"ŋ", "ng"}
};

const std::map<std::string, std::string> LIAPHON_TO_IPA = []() {
    std::map<std::string, std::string> inv;
    for (const auto& pair : IPA_TO_LIAPHON) {
        inv[pair.second] = pair.first;
    }
    return inv;
}();

std::string liaphon_to_ipa(const std::vector<std::string>& liaphon) {
    std::string ipa;
    for (const auto& phone : liaphon) {
        auto it = LIAPHON_TO_IPA.find(phone);
        if (it != LIAPHON_TO_IPA.end()) {
            ipa += it->second;
        } else {
            ipa += phone;
        }
    }
    return ipa;
}

std::vector<std::string> ipa_to_liaphon(const std::string& ipa) {
    std::vector<std::string> liaphon;
    // Simple character-by-character mapping (may need refinement for multi-char IPA)
    for (char c : ipa) {
        std::string s(1, c);
        auto it = IPA_TO_LIAPHON.find(s);
        if (it != IPA_TO_LIAPHON.end()) {
            liaphon.push_back(it->second);
        } else {
            liaphon.push_back(s);
        }
    }
    return liaphon;
}

//=============================================================================
// CTCDecoder Implementation
//=============================================================================

CTCDecoder::CTCDecoder(const DecoderConfig& config)
    : config_(config), blank_idx_(-1), sil_idx_(-1), unk_idx_(-1) {}

CTCDecoder::~CTCDecoder() = default;

bool CTCDecoder::initialize() {
    std::cout << "Initializing CTC Decoder..." << std::endl;
    
    // Load tokens
    if (!load_tokens()) {
        std::cerr << "Failed to load tokens" << std::endl;
        return false;
    }
    
    // Load lexicon
    if (!config_.lexicon_path.empty() && !load_lexicon()) {
        std::cerr << "Failed to load lexicon" << std::endl;
        return false;
    }
    
    // Load language model
    if (!config_.lm_path.empty() && !load_lm()) {
        std::cerr << "Failed to load language model" << std::endl;
        return false;
    }
    
    // Build trie for lexicon-based decoding
    if (!config_.lexicon_path.empty() && !build_trie()) {
        std::cerr << "Failed to build trie" << std::endl;
        return false;
    }
    
    // Create decoder
    using namespace fl::lib::text;
    
    if (!config_.lexicon_path.empty()) {
        // Lexicon-based decoder
        LexiconDecoderOptions options;
        options.beamSize = config_.beam_size;
        options.beamSizeToken = (config_.beam_size_token > 0) 
            ? config_.beam_size_token 
            : tokens_dict_->indexSize();
        options.beamThreshold = config_.beam_threshold;
        options.lmWeight = config_.lm_weight;
        options.wordScore = config_.word_score;
        options.unkScore = config_.unk_score;
        options.silScore = config_.sil_score;
        options.logAdd = config_.log_add;
        options.criterionType = CriterionType::CTC;
        
        // Create KenLM wrapper
        auto lm = std::make_shared<KenLM>(config_.lm_path, *word_dict_);

        lexicon_decoder_ = std::make_unique<LexiconDecoder>(
            options,
            trie_,
            lm,
            sil_idx_,
            blank_idx_,
            unk_idx_,
            std::vector<float>(),  // transitions (empty for CTC)
            false  // isLabelUnitToken
        );
    }
    
    std::cout << "CTC Decoder initialized successfully!" << std::endl;
    std::cout << "  Vocabulary size: " << get_vocab_size() << std::endl;
    std::cout << "  Blank index: " << blank_idx_ << std::endl;
    std::cout << "  Silence index: " << sil_idx_ << std::endl;
    
    return true;
}

bool CTCDecoder::load_tokens() {
    try {
        std::ifstream vocab_stream(config_.tokens_path);
        if (!vocab_stream.is_open()) {
            std::cerr << "Error loading tokens: unable to open file "
                      << config_.tokens_path << std::endl;
            return false;
        }

        std::vector<std::string> vocabulary;
        std::string line;
        auto trim = [](std::string& s) {
            auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
            s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        };

        while (std::getline(vocab_stream, line)) {
            if (line.empty()) {
                continue;
            }
            std::string token = line;
            auto sep_pos = token.find_first_of(",;\t\r");
            if (sep_pos != std::string::npos) {
                token = token.substr(0, sep_pos);
            }
            trim(token);
            if (token.empty()) {
                continue;
            }
            if (std::find(vocabulary.begin(), vocabulary.end(), token) == vocabulary.end()) {
                vocabulary.push_back(token);
            }
        }

        // Ensure special tokens match Python pipeline behaviour
        const std::vector<std::string> special_tokens = {
            "<BLANK>", "<UNK>", "<SOS>", "<EOS>", "<PAD>"
        };
        for (auto it = special_tokens.rbegin(); it != special_tokens.rend(); ++it) {
            const std::string& token = *it;
            if (std::find(vocabulary.begin(), vocabulary.end(), token) == vocabulary.end()) {
                vocabulary.insert(vocabulary.begin(), token);
            }
        }

        // Guarantee <BLANK> is at index 0
        if (vocabulary.empty()) {
            vocabulary.push_back("<BLANK>");
        } else if (vocabulary.front() != "<BLANK>") {
            auto blank_pos = std::find(vocabulary.begin(), vocabulary.end(), "<BLANK>");
            if (blank_pos != vocabulary.end()) {
                vocabulary.erase(blank_pos);
            }
            vocabulary.insert(vocabulary.begin(), "<BLANK>");
        }

        tokens_dict_ = std::make_unique<fl::lib::text::Dictionary>(vocabulary);
        
        // Build token mappings
        for (int i = 0; i < tokens_dict_->indexSize(); ++i) {
            std::string token = tokens_dict_->getEntry(i);
            token_to_index_[token] = i;
            index_to_token_[i] = token;
        }
        
        // Get special token indices
        blank_idx_ = token_to_idx(config_.blank_token);
        sil_idx_ = token_to_idx(config_.sil_token);
        unk_idx_ = token_to_idx(config_.unk_word);

        if (tokens_dict_) {
            int default_idx = blank_idx_ >= 0 ? blank_idx_ : (unk_idx_ >= 0 ? unk_idx_ : 0);
            tokens_dict_->setDefaultIndex(default_idx);
        }
        
        if (blank_idx_ < 0) {
            std::cerr << "Warning: Blank token '" << config_.blank_token 
                     << "' not found in vocabulary" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading tokens: " << e.what() << std::endl;
        return false;
    }
}

bool CTCDecoder::load_lexicon() {
    try {
        // Load word dictionary from lexicon
        auto lexicon = fl::lib::text::loadWords(config_.lexicon_path);
        word_dict_ = std::make_unique<fl::lib::text::Dictionary>(
            fl::lib::text::createWordDict(lexicon)
        );
        
        std::cout << "Loaded lexicon with " << word_dict_->indexSize() 
                 << " words" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading lexicon: " << e.what() << std::endl;
        return false;
    }
}

bool CTCDecoder::load_lm() {
    try {
        // KenLM model is loaded by flashlight's KenLM wrapper
        // We just verify the file exists
        std::ifstream lm_file(config_.lm_path);
        if (!lm_file.good()) {
            std::cerr << "LM file not found: " << config_.lm_path << std::endl;
            return false;
        }
        
        std::cout << "Language model file found: " << config_.lm_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading LM: " << e.what() << std::endl;
        return false;
    }
}

bool CTCDecoder::build_trie() {
    try {
        using namespace fl::lib::text;
        
        // Load lexicon
        auto lexicon = loadWords(config_.lexicon_path);
        
        // Create trie
        trie_ = std::make_shared<Trie>(tokens_dict_->indexSize(), sil_idx_);
        
        // Create a temporary KenLM for scoring
        auto temp_lm = std::make_shared<KenLM>(config_.lm_path, *word_dict_);
        auto start_state = temp_lm->start(false);
        
        // Insert words into trie
        for (const auto& [word, spellings] : lexicon) {
            int word_idx = word_dict_->getIndex(word);
            auto [lm_state, score] = temp_lm->score(start_state, word_idx);
            
            for (const auto& spelling : spellings) {
                std::vector<int> spelling_idxs;
                for (const auto& token : spelling) {
                    int token_idx = tokens_dict_->getIndex(token);
                    if (token_idx < 0) {
                        std::cerr << "Lexicon token '" << token
                                  << "' not found in vocabulary" << std::endl;
                        spelling_idxs.clear();
                        break;
                    }
                    spelling_idxs.push_back(token_idx);
                }
                if (!spelling_idxs.empty()) {
                    trie_->insert(spelling_idxs, word_idx, score);
                }
            }
        }
        
        // Smear the trie
        trie_->smear(SmearingMode::MAX);
        
        std::cout << "Trie built successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building trie: " << e.what() << std::endl;
        return false;
    }
}

void CTCDecoder::log_softmax(const float* logits, float* log_probs, int T, int V) {
    for (int t = 0; t < T; ++t) {
        const float* logit_row = logits + t * V;
        float* log_prob_row = log_probs + t * V;
        
        // Find max for numerical stability
        float max_logit = *std::max_element(logit_row, logit_row + V);
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int v = 0; v < V; ++v) {
            log_prob_row[v] = std::exp(logit_row[v] - max_logit);
            sum_exp += log_prob_row[v];
        }
        
        // Convert to log probabilities
        float log_sum = std::log(sum_exp);
        for (int v = 0; v < V; ++v) {
            log_prob_row[v] = logit_row[v] - max_logit - log_sum;
        }
    }
}

std::vector<CTCHypothesis> CTCDecoder::decode(const float* logits, int T, int V) {
    // Apply log softmax
    std::vector<float> log_probs(T * V);
    log_softmax(logits, log_probs.data(), T, V);
    
    return decode_log_probs(log_probs.data(), T, V);
}

std::vector<CTCHypothesis> CTCDecoder::decode_log_probs(const float* log_probs, int T, int V) {
    std::vector<CTCHypothesis> results;
    
    if (!lexicon_decoder_) {
        std::cerr << "Decoder not initialized" << std::endl;
        return results;
    }
    
    try {
        // Run decoder
        auto decoder_results = lexicon_decoder_->decode(log_probs, T, V);
        
        // Convert results to our format
        for (const auto& result : decoder_results) {
            CTCHypothesis hyp;
            hyp.tokens = result.tokens;
            hyp.score = result.score;
            
            // Convert word indices to strings
            for (int word_idx : result.words) {
                if (word_idx >= 0 && word_idx < static_cast<int>(word_dict_->indexSize())) {
                    hyp.words.push_back(word_dict_->getEntry(word_idx));
                }
            }
            
            results.push_back(hyp);
            
            if (results.size() >= config_.nbest) {
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Decoding error: " << e.what() << std::endl;
    }
    
    return results;
}

std::vector<std::string> CTCDecoder::idxs_to_tokens(const std::vector<int>& indices) {
    std::vector<std::string> tokens;
    tokens.reserve(indices.size());

    for (int idx : indices) {
        tokens.push_back(idx_to_token(idx));
    }

    if (tokens.size() >= 2) {
        tokens.erase(tokens.begin());
        tokens.pop_back();
    }

    std::vector<std::string> filtered;
    filtered.reserve(tokens.size());
    for (const auto& token : tokens) {
        if (token.empty()) {
            continue;
        }
        if (token == "<BLANK>" || token == "<PAD>" ||
            token == "<SOS>" || token == "<EOS>") {
            continue;
        }
        filtered.push_back(token);
    }

    std::vector<std::string> deduped;
    deduped.reserve(filtered.size());
    for (const auto& token : filtered) {
        if (deduped.empty() || deduped.back() != token) {
            deduped.push_back(token);
        }
    }

    while (!deduped.empty() && deduped.back() == "_") {
        deduped.pop_back();
    }

    return deduped;
}

int CTCDecoder::get_vocab_size() const {
    return tokens_dict_ ? tokens_dict_->indexSize() : 0;
}

int CTCDecoder::token_to_idx(const std::string& token) const {
    auto it = token_to_index_.find(token);
    if (it != token_to_index_.end()) {
        return it->second;
    }
    return -1;
}

std::string CTCDecoder::idx_to_token(int idx) const {
    auto it = index_to_token_.find(idx);
    if (it != index_to_token_.end()) {
        return it->second;
    }
    return "";
}

//=============================================================================
// FeatureExtractor Implementation
//=============================================================================

FeatureExtractor::FeatureExtractor() {}

float FeatureExtractor::scalar_distance(float x1, float y1, float z1,
                                       float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float FeatureExtractor::polygon_area(const std::vector<float>& xs,
                                    const std::vector<float>& ys) {
    if (xs.size() != ys.size() || xs.empty()) {
        return 0.0f;
    }
    
    float area = 0.0f;
    int n = xs.size();
    
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += xs[i] * ys[j];
        area -= xs[j] * ys[i];
    }
    
    return std::abs(area) * 0.5f;
}

float FeatureExtractor::mean_contour_curvature(
    const std::vector<std::pair<float, float>>& points) {
    
    if (points.size() < 3) {
        return 0.0f;
    }
    
    std::vector<float> angles;
    int n = points.size();
    
    for (int i = 0; i < n; ++i) {
        auto p_prev = points[(i - 1 + n) % n];
        auto p_curr = points[i];
        auto p_next = points[(i + 1) % n];
        
        float v1x = p_prev.first - p_curr.first;
        float v1y = p_prev.second - p_curr.second;
        float v2x = p_next.first - p_curr.first;
        float v2y = p_next.second - p_curr.second;
        
        float norm1 = std::sqrt(v1x*v1x + v1y*v1y);
        float norm2 = std::sqrt(v2x*v2x + v2y*v2y);
        
        if (norm1 < 1e-6f || norm2 < 1e-6f) {
            continue;
        }
        
        float cosang = (v1x*v2x + v1y*v2y) / (norm1 * norm2);
        cosang = std::max(-1.0f, std::min(1.0f, cosang));
        angles.push_back(std::acos(cosang));
    }
    
    if (angles.empty()) {
        return 0.0f;
    }
    
    return std::accumulate(angles.begin(), angles.end(), 0.0f) / angles.size();
}

float FeatureExtractor::get_angle(float x1, float y1, float z1,
                                 float x2, float y2, float z2,
                                 float x3, float y3, float z3) {
    float v1x = x1 - x2;
    float v1y = y1 - y2;
    float v1z = z1 - z2;
    float v2x = x3 - x2;
    float v2y = y3 - y2;
    float v2z = z3 - z2;
    
    float dot = v1x*v2x + v1y*v2y + v1z*v2z;
    float norm1 = std::sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
    float norm2 = std::sqrt(v2x*v2x + v2y*v2y + v2z*v2z);
    
    if (norm1 < 1e-6f || norm2 < 1e-6f) {
        return 0.0f;
    }
    
    float cosang = dot / (norm1 * norm2);
    cosang = std::max(-1.0f, std::min(1.0f, cosang));
    return std::acos(cosang);
}

FrameFeatures FeatureExtractor::extract(
    const LandmarkResults& landmarks,
    const LandmarkResults* prev_landmarks,
    const LandmarkResults* prev2_landmarks) {
    FrameFeatures invalid;

    const auto get_face = [](const LandmarkResults& data, int idx, float& x, float& y, float& z) {
        if (idx < 0 || idx >= static_cast<int>(data.face_landmarks.size())) {
            return false;
        }
        const auto& lm = data.face_landmarks[idx];
        if (!std::isfinite(lm.x) || !std::isfinite(lm.y) || !std::isfinite(lm.z)) {
            return false;
        }
        x = lm.x;
        y = lm.y;
        z = lm.z;
        return true;
    };

    const auto get_hand = [](const LandmarkResults& data, int idx, float& x, float& y, float& z) {
        if (idx < 0 || idx >= static_cast<int>(data.hand_landmarks.size())) {
            return false;
        }
        const auto& lm = data.hand_landmarks[idx];
        if (!std::isfinite(lm.x) || !std::isfinite(lm.y) || !std::isfinite(lm.z)) {
            return false;
        }
        x = lm.x;
        y = lm.y;
        z = lm.z;
        return true;
    };

    // Normalization factors
    float f1x, f1y, f1z, f2x, f2y, f2z;
    if (!get_face(landmarks, 454, f1x, f1y, f1z) ||
        !get_face(landmarks, 234, f2x, f2y, f2z)) {
        return invalid;
    }

    float face_width = scalar_distance(f1x, f1y, f1z, f2x, f2y, f2z);
    if (face_width <= 1e-6f) {
        return invalid;
    }

    float h0x, h0y, h0z, h9x, h9y, h9z;
    float hand_span = face_width;
    if (get_hand(landmarks, 0, h0x, h0y, h0z) && get_hand(landmarks, 9, h9x, h9y, h9z)) {
        hand_span = scalar_distance(h0x, h0y, h0z, h9x, h9y, h9z);
        if (hand_span <= 1e-6f) {
            hand_span = face_width;
        }
    }

    // Hand-face distances & angles (hand position features)
    std::vector<float> hand_position_features;
    hand_position_features.reserve(18);

    static const std::array<int, 3> kHandIndices = {8, 9, 12};
    static const std::array<int, 5> kFaceIndices = {234, 200, 214, 454, 280};

    for (int hand_idx : kHandIndices) {
        float hx, hy, hz;
        if (!get_hand(landmarks, hand_idx, hx, hy, hz)) {
            return invalid;
        }

        for (int face_idx : kFaceIndices) {
            float fx, fy, fz;
            if (!get_face(landmarks, face_idx, fx, fy, fz)) {
                return invalid;
            }

            float dist = scalar_distance(hx, hy, hz, fx, fy, fz) / face_width;
            hand_position_features.push_back(dist);

            if (face_idx == 200) {
                float dx = (fx - hx) / face_width;
                float dy = (fy - hy) / face_width;
                hand_position_features.push_back(std::atan2(dy, dx));
            }
        }
    }

    if (hand_position_features.size() != 18) {
        return invalid;
    }

    // Hand-hand distances (hand shape features)
    std::vector<float> hand_shape_features;
    hand_shape_features.reserve(7);

    static const std::array<std::pair<int, int>, 5> kHandShapePairs = {
        std::make_pair(0, 4), std::make_pair(0, 8), std::make_pair(0, 12),
        std::make_pair(0, 16), std::make_pair(0, 20)
    };

    for (const auto& pair : kHandShapePairs) {
        float x1, y1, z1, x2, y2, z2;
        if (!get_hand(landmarks, pair.first, x1, y1, z1) ||
            !get_hand(landmarks, pair.second, x2, y2, z2)) {
            return invalid;
        }
        float dist = scalar_distance(x1, y1, z1, x2, y2, z2) / hand_span;
        hand_shape_features.push_back(dist);
    }

    // Lip metrics
    std::vector<float> lip_features;
    lip_features.reserve(8);

    float lx61, ly61, lz61, lx291, ly291, lz291;
    if (!get_face(landmarks, 61, lx61, ly61, lz61) ||
        !get_face(landmarks, 291, lx291, ly291, lz291)) {
        return invalid;
    }
    lip_features.push_back(
        scalar_distance(lx61, ly61, lz61, lx291, ly291, lz291) / face_width);

    float lx0, ly0, lz0, lx17, ly17, lz17;
    if (!get_face(landmarks, 0, lx0, ly0, lz0) ||
        !get_face(landmarks, 17, lx17, ly17, lz17)) {
        return invalid;
    }
    lip_features.push_back(
        scalar_distance(lx0, ly0, lz0, lx17, ly17, lz17) / face_width);

    static const std::array<int, 20> kLipOuter = {
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146
    };

    std::vector<float> lip_xs;
    std::vector<float> lip_ys;
    lip_xs.reserve(kLipOuter.size());
    lip_ys.reserve(kLipOuter.size());
    std::vector<std::pair<float, float>> lip_points;
    lip_points.reserve(kLipOuter.size());

    for (int idx : kLipOuter) {
        float x, y, z;
        if (!get_face(landmarks, idx, x, y, z)) {
            return invalid;
        }
        lip_xs.push_back(x);
        lip_ys.push_back(y);
        lip_points.emplace_back(x, y);
    }

    float lip_area = polygon_area(lip_xs, lip_ys) / (face_width * face_width);
    lip_features.push_back(lip_area);
    lip_features.push_back(mean_contour_curvature(lip_points));

    // Motion features require previous frames
    if (prev_landmarks == nullptr || prev2_landmarks == nullptr) {
        return invalid;
    }

    float prev_lx0, prev_ly0, prev_lz0;
    float prev2_lx0, prev2_ly0, prev2_lz0;
    if (!get_face(*prev_landmarks, 0, prev_lx0, prev_ly0, prev_lz0) ||
        !get_face(*prev2_landmarks, 0, prev2_lx0, prev2_ly0, prev2_lz0)) {
        return invalid;
    }

    float lip_vel_x = (lx0 - prev_lx0) / face_width;
    float lip_vel_y = (ly0 - prev_ly0) / face_width;
    lip_features.push_back(lip_vel_x);
    lip_features.push_back(lip_vel_y);

    float prev_vel_x = (prev_lx0 - prev2_lx0) / face_width;
    float prev_vel_y = (prev_ly0 - prev2_ly0) / face_width;
    lip_features.push_back(lip_vel_x - prev_vel_x);
    lip_features.push_back(lip_vel_y - prev_vel_y);

    // Hand velocity features
    float hx8, hy8, hz8, prev_hx8, prev_hy8, prev_hz8;
    if (!get_hand(landmarks, 8, hx8, hy8, hz8) ||
        !get_hand(*prev_landmarks, 8, prev_hx8, prev_hy8, prev_hz8)) {
        return invalid;
    }
    hand_shape_features.push_back((hx8 - prev_hx8) / hand_span);
    hand_shape_features.push_back((hy8 - prev_hy8) / hand_span);

    if (hand_shape_features.size() != 7 ||
        hand_position_features.size() != 18 ||
        lip_features.size() != 8) {
        return invalid;
    }

    FrameFeatures features;
    features.hand_shape = std::move(hand_shape_features);
    features.hand_position = std::move(hand_position_features);
    features.lips = std::move(lip_features);
    return features;
}

//=============================================================================
// WindowProcessor Implementation
//=============================================================================

WindowProcessor::WindowProcessor(CTCDecoder* decoder, TFLiteSequenceModel* sequence_model)
    : decoder_(decoder),
      sequence_model_(sequence_model),
      chunk_idx_(0),
      next_window_needed_(WINDOW_SIZE),
      frame_count_(0),
      effective_vocab_size_(decoder ? decoder->get_vocab_size() : 0),
      total_frames_seen_(0),
      chunks_processed_(0) {}

void WindowProcessor::reset() {
    valid_features_.clear();
    all_logits_.clear();
    chunk_idx_ = 0;
    next_window_needed_ = WINDOW_SIZE;
    frame_count_ = 0;
    effective_vocab_size_ = decoder_ ? decoder_->get_vocab_size() : 0;
    total_frames_seen_ = 0;
    chunks_processed_ = 0;
}

bool WindowProcessor::push_frame(const FrameFeatures& features) {
    total_frames_seen_++;

    if (!features.is_valid()) {
        return false;
    }
    
    valid_features_.push_back(features);
    frame_count_++;
    
    return valid_features_.size() >= next_window_needed_;
}

RecognitionResult WindowProcessor::process_window() {
    RecognitionResult result;
    result.frame_number = frame_count_;
    result.confidence = 0.0f;

    if (!sequence_model_ || !sequence_model_->is_loaded()) {
        return result;
    }

    const int num_valid = static_cast<int>(valid_features_.size());
    if (num_valid < next_window_needed_) {
        return result;
    }

    int window_start = 0;
    int window_end = 0;
    int commit_start = 0;
    int commit_end = 0;

    if (chunk_idx_ == 0) {
        window_start = 0;
        window_end = std::min(WINDOW_SIZE - 1, num_valid - 1);
        commit_start = 0;
        commit_end = std::min(COMMIT_SIZE - 1, num_valid - 1);
        next_window_needed_ = LEFT_CONTEXT + WINDOW_SIZE;
    } else if (chunk_idx_ == 1) {
        window_start = LEFT_CONTEXT;
        window_end = std::min(window_start + WINDOW_SIZE - 1, num_valid - 1);
        commit_start = COMMIT_SIZE;
        commit_end = std::min(commit_start + LEFT_CONTEXT - 1, num_valid - 1);
        next_window_needed_ = COMMIT_SIZE + WINDOW_SIZE;
    } else {
        window_start = COMMIT_SIZE * (chunk_idx_ - 1);
        window_end = std::min(window_start + WINDOW_SIZE - 1, num_valid - 1);
        commit_start = window_start + LEFT_CONTEXT;
        commit_end = std::min(commit_start + COMMIT_SIZE - 1, num_valid - 1);
        next_window_needed_ = COMMIT_SIZE * chunk_idx_ + WINDOW_SIZE;
    }

    std::cout << "[Valid frames: " << num_valid << "] Chunk " << chunk_idx_
              << ": window=[" << window_start << ", " << window_end
              << "], commit=[" << commit_start << ", " << commit_end << "]" << std::endl;

    int window_vocab_size = 0;
    auto committed_logits = process_single_window(
        window_start,
        window_end,
        commit_start,
        commit_end,
        window_vocab_size);

    if (committed_logits.empty()) {
        chunk_idx_++;
        return result;
    }

    if (window_vocab_size > 0) {
        if (effective_vocab_size_ <= 0) {
            effective_vocab_size_ = window_vocab_size;
        } else if (effective_vocab_size_ != window_vocab_size) {
            effective_vocab_size_ = window_vocab_size;
        }
    }

    if (effective_vocab_size_ <= 0) {
        chunk_idx_++;
        return result;
    }

    all_logits_.push_back(std::move(committed_logits));

    int vocab_size = decoder_ ? decoder_->get_vocab_size() : 0;
    if (vocab_size <= 0) {
        vocab_size = effective_vocab_size_;
    }

    if (vocab_size <= 0) {
        chunk_idx_++;
        return result;
    }

    int total_frames = 0;
    for (const auto& logits : all_logits_) {
        if (!logits.empty()) {
            total_frames += static_cast<int>(logits.size() / vocab_size);
        }
    }

    if (total_frames <= 0) {
        chunk_idx_++;
        return result;
    }

    std::vector<float> full_logits;
    full_logits.reserve(static_cast<size_t>(total_frames) * vocab_size);
    for (const auto& logits : all_logits_) {
        full_logits.insert(full_logits.end(), logits.begin(), logits.end());
    }

    std::cout << "  Full accumulated logits shape: [" << total_frames
              << " x " << vocab_size << "]" << std::endl;

    auto hypotheses = decoder_->decode(full_logits.data(), total_frames, vocab_size);
    if (!hypotheses.empty()) {
        result.phonemes = decoder_->idxs_to_tokens(hypotheses[0].tokens);
        result.confidence = hypotheses[0].score;

        std::cout << "  Decoded sentence after chunk " << chunk_idx_ << ": ";
        for (const auto& token : result.phonemes) {
            std::cout << token << ' ';
        }
        std::cout << std::endl;

        ++chunks_processed_;
    }

    chunk_idx_++;
    return result;
}

std::vector<float> WindowProcessor::process_single_window(
    int window_start,
    int window_end,
    int commit_start,
    int commit_end,
    int& out_vocab_size) {

    out_vocab_size = 0;

    if (!sequence_model_ || !sequence_model_->is_loaded() || window_end < window_start) {
        return {};
    }

    const int window_size_actual = window_end - window_start + 1;
    if (window_size_actual <= 0) {
        return {};
    }

    std::vector<FrameFeatures> padded_features;
    padded_features.reserve(WINDOW_SIZE);
    for (int idx = window_start; idx <= window_end; ++idx) {
        padded_features.push_back(valid_features_[idx]);
    }
    if (static_cast<int>(padded_features.size()) < WINDOW_SIZE) {
        FrameFeatures zero;
        zero.hand_shape.assign(7, 0.0f);
        zero.hand_position.assign(18, 0.0f);
        zero.lips.assign(8, 0.0f);
        padded_features.resize(WINDOW_SIZE, zero);
    }

    auto window_logits = sequence_model_->infer(padded_features, WINDOW_SIZE);
    out_vocab_size = sequence_model_->vocab_size();
    const int seq_len = sequence_model_->last_sequence_length();

    if (window_logits.empty() || out_vocab_size <= 0 || seq_len <= 0) {
        return {};
    }

    int commit_start_rel = commit_start - window_start;
    int commit_end_rel = commit_end - window_start;
    commit_start_rel = std::max(commit_start_rel, 0);
    commit_end_rel = std::min(commit_end_rel, seq_len - 1);

    if (commit_start_rel > commit_end_rel) {
        return {};
    }

    std::vector<float> committed_logits;
    committed_logits.reserve(static_cast<size_t>(commit_end_rel - commit_start_rel + 1) * out_vocab_size);
    for (int t = commit_start_rel; t <= commit_end_rel; ++t) {
        const float* row = window_logits.data() + static_cast<size_t>(t) * out_vocab_size;
        committed_logits.insert(committed_logits.end(), row, row + out_vocab_size);
    }

    return committed_logits;
}

RecognitionResult WindowProcessor::finalize() {
    RecognitionResult result;
    result.frame_number = frame_count_;
    result.confidence = 0.0f;

    if (!sequence_model_ || !sequence_model_->is_loaded()) {
        return result;
    }

    const int num_valid = static_cast<int>(valid_features_.size());
    if (num_valid == 0) {
        return result;
    }

    int frames_committed = 0;
    if (chunk_idx_ == 0) {
        frames_committed = 0;
    } else if (chunk_idx_ == 1) {
        frames_committed = COMMIT_SIZE;
    } else {
        frames_committed = COMMIT_SIZE + LEFT_CONTEXT + (chunk_idx_ - 2) * COMMIT_SIZE;
    }

    if (frames_committed >= num_valid) {
        return result;
    }

    int window_start = 0;
    int window_end = num_valid - 1;
    int commit_start = 0;
    int commit_end = num_valid - 1;

    if (chunk_idx_ == 0) {
        window_start = 0;
        window_end = num_valid - 1;
        commit_start = 0;
        commit_end = num_valid - 1;
    } else if (chunk_idx_ == 1) {
        window_start = LEFT_CONTEXT;
        window_end = num_valid - 1;
        commit_start = COMMIT_SIZE;
        commit_end = num_valid - 1;
    } else {
        window_start = COMMIT_SIZE * (chunk_idx_ - 1);
        window_end = num_valid - 1;
        commit_start = window_start + LEFT_CONTEXT;
        commit_end = num_valid - 1;
    }

    if (window_end - window_start + 1 < LEFT_CONTEXT) {
        return result;
    }

    int window_vocab_size = 0;
    auto committed_logits = process_single_window(
        window_start,
        window_end,
        commit_start,
        commit_end,
        window_vocab_size);

    if (committed_logits.empty()) {
        return result;
    }

    if (window_vocab_size > 0) {
        if (effective_vocab_size_ <= 0) {
            effective_vocab_size_ = window_vocab_size;
        } else if (effective_vocab_size_ != window_vocab_size) {
            effective_vocab_size_ = window_vocab_size;
        }
    }

    if (effective_vocab_size_ <= 0) {
        return result;
    }

    all_logits_.push_back(std::move(committed_logits));

    int vocab_size = decoder_ ? decoder_->get_vocab_size() : 0;
    if (vocab_size <= 0) {
        vocab_size = effective_vocab_size_;
    }

    if (vocab_size <= 0) {
        return result;
    }

    int total_frames = 0;
    for (const auto& logits : all_logits_) {
        if (!logits.empty()) {
            total_frames += static_cast<int>(logits.size() / vocab_size);
        }
    }

    if (total_frames <= 0) {
        return result;
    }

    std::vector<float> full_logits;
    full_logits.reserve(static_cast<size_t>(total_frames) * vocab_size);
    for (const auto& logits : all_logits_) {
        full_logits.insert(full_logits.end(), logits.begin(), logits.end());
    }

    auto hypotheses = decoder_->decode(full_logits.data(), total_frames, vocab_size);
    if (!hypotheses.empty()) {
        result.phonemes = decoder_->idxs_to_tokens(hypotheses[0].tokens);
        result.confidence = hypotheses[0].score;

        ++chunks_processed_;
    }

    return result;
}

int WindowProcessor::valid_frame_count() const {
    return static_cast<int>(valid_features_.size());
}

int WindowProcessor::total_frames_seen() const {
    return total_frames_seen_;
}

int WindowProcessor::dropped_frame_count() const {
    return total_frames_seen_ - static_cast<int>(valid_features_.size());
}

int WindowProcessor::chunks_processed() const {
    return chunks_processed_;
}

//=============================================================================
// SentenceCorrector Implementation
//=============================================================================

SentenceCorrector::SentenceCorrector(const std::string& homophones_path,
                                   const std::string& kenlm_path)
    : homophones_path_(homophones_path), kenlm_path_(kenlm_path) {}

namespace {
bool parse_homophone_line(const std::string& line,
                          std::string& ipa,
                          std::vector<std::string>& words) {
    ipa.clear();
    words.clear();

    auto find_value = [](const std::string& src, const std::string& key) -> std::string {
        size_t pos = src.find(key);
        if (pos == std::string::npos) {
            return {};
        }
        pos = src.find('"', pos + key.size());
        if (pos == std::string::npos) {
            return {};
        }
        size_t end = src.find('"', pos + 1);
        if (end == std::string::npos) {
            return {};
        }
        return src.substr(pos + 1, end - pos - 1);
    };

    ipa = find_value(line, "\"ipa\"");
    if (ipa.empty()) {
        return false;
    }

    size_t words_pos = line.find("\"words\"");
    if (words_pos != std::string::npos) {
        size_t start = line.find('[', words_pos);
        size_t end = line.find(']', start);
        if (start != std::string::npos && end != std::string::npos && end > start) {
            std::string array_str = line.substr(start + 1, end - start - 1);
            size_t pos = 0;
            while (true) {
                size_t q1 = array_str.find('"', pos);
                if (q1 == std::string::npos) {
                    break;
                }
                size_t q2 = array_str.find('"', q1 + 1);
                if (q2 == std::string::npos) {
                    break;
                }
                words.emplace_back(array_str.substr(q1 + 1, q2 - q1 - 1));
                pos = q2 + 1;
            }
        }
    }

    if (words.empty()) {
        words.push_back(ipa);
    }

    return true;
}

std::string capitalize_sentence(const std::string& text) {
    if (text.empty()) {
        return text;
    }
    std::string result = text;
    result[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(result[0])));
    return result;
}
} // namespace

bool SentenceCorrector::initialize() {
    ipa_to_homophones_.clear();
    kenlm_model_.reset();

    std::ifstream file(homophones_path_);
    if (!file.is_open()) {
        std::cerr << "Failed to open homophones file: " << homophones_path_ << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::string ipa;
        std::vector<std::string> words;
        if (parse_homophone_line(line, ipa, words)) {
            ipa_to_homophones_[ipa] = words;
        }
    }

    try {
        kenlm_model_ = std::make_unique<lm::ngram::Model>(kenlm_path_.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load KenLM model: " << e.what() << std::endl;
        return false;
    }

    return true;
}

std::string SentenceCorrector::correct(const std::vector<std::string>& liaphon_phonemes) {
    if (!kenlm_model_) {
        return {};
    }

    std::string ipa_sentence;
    ipa_sentence.reserve(liaphon_phonemes.size() * 2);
    for (const auto& phone : liaphon_phonemes) {
        auto it = LIAPHON_TO_IPA.find(phone);
        if (it != LIAPHON_TO_IPA.end()) {
            ipa_sentence += it->second;
        } else {
            ipa_sentence += phone;
        }
    }

    std::vector<std::string> ipa_tokens;
    std::string current;
    for (char ch : ipa_sentence) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                ipa_tokens.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        ipa_tokens.push_back(current);
    }

    if (ipa_tokens.empty() && !ipa_sentence.empty()) {
        ipa_tokens.push_back(ipa_sentence);
    }

    std::vector<std::vector<std::string>> homophone_lists;
    homophone_lists.reserve(ipa_tokens.size());
    for (const auto& token : ipa_tokens) {
        auto it = ipa_to_homophones_.find(token);
        if (it != ipa_to_homophones_.end() && !it->second.empty()) {
            homophone_lists.push_back(it->second);
        } else {
            homophone_lists.push_back({token});
        }
    }

    if (homophone_lists.empty()) {
        return {};
    }

    auto best_sequence = beam_search(homophone_lists, 20);
    if (best_sequence.empty()) {
        return {};
    }

    std::string sentence;
    for (size_t i = 0; i < best_sequence.size(); ++i) {
        if (i > 0) {
            sentence.push_back(' ');
        }
        sentence.append(best_sequence[i]);
    }

    sentence = capitalize_sentence(sentence);
    if (!sentence.empty() && sentence.back() != '.') {
        sentence.push_back('.');
    }

    return sentence;
}

std::vector<std::string> SentenceCorrector::beam_search(
    const std::vector<std::vector<std::string>>& homophone_lists,
    int beam_width) {

    if (!kenlm_model_) {
        return {};
    }

    struct Beam {
        double score;
        lm::ngram::State state;
        std::vector<std::string> words;
    };

    std::vector<Beam> beams;
    beams.reserve(beam_width);
    lm::ngram::State start_state = kenlm_model_->BeginSentenceState();
    beams.push_back({0.0, start_state, {}});

    for (const auto& homophones : homophone_lists) {
        std::vector<Beam> new_beams;
        for (const auto& beam : beams) {
            for (const auto& word : homophones) {
                lm::WordIndex idx = kenlm_model_->GetVocabulary().Index(word);
                lm::ngram::State out_state;
                double score = kenlm_model_->BaseScore(&beam.state, idx, &out_state);
                Beam next;
                next.score = beam.score + score;
                next.state = out_state;
                next.words = beam.words;
                next.words.push_back(word);
                new_beams.push_back(std::move(next));
            }
        }

        if (new_beams.empty()) {
            return {};
        }

        std::sort(new_beams.begin(), new_beams.end(),
                  [](const Beam& a, const Beam& b) { return a.score > b.score; });

        if (static_cast<int>(new_beams.size()) > beam_width) {
            new_beams.resize(beam_width);
        }

        beams.swap(new_beams);
    }

    if (beams.empty()) {
        return {};
    }

    return beams.front().words;
}

namespace {
std::string remove_accents(const std::string& input) {
    static const std::unordered_map<std::string, std::string> replacements = {
        {"\xC3\x80", "A"}, {"\xC3\x81", "A"}, {"\xC3\x82", "A"}, {"\xC3\x83", "A"},
        {"\xC3\x84", "A"}, {"\xC3\x87", "C"}, {"\xC3\x88", "E"}, {"\xC3\x89", "E"},
        {"\xC3\x8A", "E"}, {"\xC3\x8B", "E"}, {"\xC3\x8E", "I"}, {"\xC3\x8F", "I"},
        {"\xC3\x94", "O"}, {"\xC3\x96", "O"}, {"\xC3\x99", "U"}, {"\xC3\x9B", "U"},
        {"\xC3\x9C", "U"}, {"\xC3\xA0", "a"}, {"\xC3\xA1", "a"}, {"\xC3\xA2", "a"},
        {"\xC3\xA3", "a"}, {"\xC3\xA4", "a"}, {"\xC3\xA7", "c"}, {"\xC3\xA8", "e"},
        {"\xC3\xA9", "e"}, {"\xC3\xAA", "e"}, {"\xC3\xAB", "e"}, {"\xC3\xAE", "i"},
        {"\xC3\xAF", "i"}, {"\xC3\xB4", "o"}, {"\xC3\xB6", "o"}, {"\xC3\xB9", "u"},
        {"\xC3\xBB", "u"}, {"\xC3\xBC", "u"}, {"\xC5\x92", "OE"}, {"\xC5\x93", "oe"}
    };

    std::string output;
    for (size_t i = 0; i < input.size();) {
        bool replaced = false;
        for (const auto& kv : replacements) {
            const auto& key = kv.first;
            if (input.compare(i, key.size(), key) == 0) {
                output.append(kv.second);
                i += key.size();
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            output.push_back(input[i]);
            ++i;
        }
    }
    return output;
}
}

bool write_subtitled_video(
    const std::string& input_path,
    const std::deque<RecognitionResult>& recognition_results,
    const std::string& output_path,
    double fps) {

    std::vector<RecognitionResult> results(recognition_results.begin(), recognition_results.end());
    std::sort(results.begin(), results.end(),
              [](const RecognitionResult& a, const RecognitionResult& b) {
                  return a.frame_number < b.frame_number;
              });

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << input_path << std::endl;
        return false;
    }

    double video_fps = fps > 0.0 ? fps : cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0.0) {
        video_fps = 30.0;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    width &= ~1;
    height &= ~1;

    cv::VideoWriter writer(
        output_path,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        video_fps,
        cv::Size(width, height));

    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter: " << output_path << std::endl;
        cap.release();
        return false;
    }

    size_t result_index = 0;
    int next_frame_update = results.empty() ? std::numeric_limits<int>::max()
                                            : results.front().frame_number;
    std::string current_text;

    int frame_num = 0;
    cv::Mat frame;
    while (cap.read(frame)) {
        ++frame_num;
        if (frame.empty()) {
            break;
        }

        if (frame_num >= next_frame_update && result_index < results.size()) {
            const auto& entry = results[result_index];
            if (!entry.french_sentence.empty()) {
                current_text = remove_accents(entry.french_sentence);
            } else if (!entry.phonemes.empty()) {
                current_text.clear();
                for (size_t i = 0; i < entry.phonemes.size(); ++i) {
                    if (i > 0) {
                        current_text.push_back(' ');
                    }
                    current_text.append(entry.phonemes[i]);
                }
            }

            ++result_index;
            next_frame_update = (result_index < results.size())
                ? results[result_index].frame_number
                : std::numeric_limits<int>::max();
        }

        if (!current_text.empty()) {
            int baseline = 0;
            const int font = cv::FONT_HERSHEY_SIMPLEX;
            double font_scale = 1.0;
            int thickness = 2;
            cv::Size text_size = cv::getTextSize(current_text, font, font_scale, thickness, &baseline);
            int x = (width - text_size.width) / 2;
            int y = static_cast<int>(height * 0.9);

            cv::putText(frame, current_text, cv::Point(x, y), font, font_scale,
                        cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
            cv::putText(frame, current_text, cv::Point(x, y), font, font_scale,
                        cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        }

        writer.write(frame);
    }

    writer.release();
    cap.release();

    return true;
}

} // namespace cued_speech

