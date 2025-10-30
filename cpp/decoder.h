/**
 * Cued Speech Decoder - C++ Implementation
 * 
 * This header provides the main interface for decoding cued speech videos
 * using TensorFlow Lite models for feature extraction and flashlight-text
 * with KenLM for CTC beam search decoding.
 */

#ifndef CUED_SPEECH_DECODER_H
#define CUED_SPEECH_DECODER_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <unordered_map>

// Forward declarations
namespace fl {
namespace lib {
namespace text {
class LexiconDecoder;
class LexiconFreeDecoder;
class Dictionary;
class Trie;
}
}
}

namespace lm {
namespace ngram {
class Model;
}
}

namespace cued_speech {

// Constants
constexpr int WINDOW_SIZE = 100;
constexpr int COMMIT_SIZE = 50;
constexpr int LEFT_CONTEXT = 25;
constexpr int RIGHT_CONTEXT = 25;

/**
 * Hypothesis returned by the decoder
 */
struct CTCHypothesis {
    std::vector<int> tokens;           // Token indices
    std::vector<std::string> words;    // Decoded words
    float score;                        // Hypothesis score
    std::vector<int> timesteps;        // Token timesteps
};

/**
 * Decoder configuration
 */
struct DecoderConfig {
    std::string lexicon_path;
    std::string tokens_path;
    std::string lm_path;              // KenLM binary path
    std::string lm_dict_path;         // Optional
    
    int nbest = 1;
    int beam_size = 40;
    int beam_size_token = -1;         // -1 means use vocab size
    float beam_threshold = 50.0f;
    float lm_weight = 3.23f;
    float word_score = 0.0f;
    float unk_score = -std::numeric_limits<float>::infinity();
    float sil_score = 0.0f;
    bool log_add = false;
    
    std::string blank_token = "<BLANK>";
    std::string sil_token = "_";
    std::string unk_word = "<UNK>";
};

/**
 * Feature extraction result for a single frame
 */
struct FrameFeatures {
    std::vector<float> hand_shape;     // 7 features
    std::vector<float> hand_position;  // 18 features
    std::vector<float> lips;           // 8 features
    
    bool is_valid() const {
        return hand_shape.size() == 7 && 
               hand_position.size() == 18 && 
               lips.size() == 8;
    }
};

/**
 * Landmark data for a single point
 */
struct Landmark {
    float x, y, z;
};

/**
 * Raw landmark results from detection models
 */
struct LandmarkResults {
    std::vector<Landmark> face_landmarks;
    std::vector<Landmark> hand_landmarks;
    std::vector<Landmark> pose_landmarks;
};

/**
 * Recognition result for a decoded segment
 */
struct RecognitionResult {
    int frame_number;
    std::vector<std::string> phonemes;
    std::string french_sentence;
    float confidence;
};

class TFLiteSequenceModel {
public:
    TFLiteSequenceModel();
    ~TFLiteSequenceModel();

    bool load(const std::string& model_path);
    std::vector<float> infer(const std::vector<FrameFeatures>& frames, int window_size);
    int vocab_size() const;
    int last_sequence_length() const;
    bool is_loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Main CTC Decoder class
 * 
 * Wraps flashlight-text lexicon decoder with KenLM language model
 */
class CTCDecoder {
public:
    /**
     * Constructor
     * 
     * @param config Decoder configuration
     */
    explicit CTCDecoder(const DecoderConfig& config);
    
    /**
     * Destructor
     */
    ~CTCDecoder();
    
    // Disable copy
    CTCDecoder(const CTCDecoder&) = delete;
    CTCDecoder& operator=(const CTCDecoder&) = delete;
    
    /**
     * Initialize the decoder (load models, lexicon, etc.)
     * 
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * Decode a batch of logits
     * 
     * @param logits 2D array [T x V] where T=time steps, V=vocab size
     * @param T Number of time steps
     * @param V Vocabulary size
     * @return Vector of hypotheses (size = nbest)
     */
    std::vector<CTCHypothesis> decode(const float* logits, int T, int V);
    
    /**
     * Decode from log probabilities
     * 
     * @param log_probs 2D array [T x V] in log space
     * @param T Number of time steps
     * @param V Vocabulary size
     * @return Vector of hypotheses (size = nbest)
     */
    std::vector<CTCHypothesis> decode_log_probs(const float* log_probs, int T, int V);
    
    /**
     * Convert token indices to token strings
     * 
     * @param indices Vector of token indices
     * @return Vector of token strings
     */
    std::vector<std::string> idxs_to_tokens(const std::vector<int>& indices);
    
    /**
     * Get vocabulary size
     */
    int get_vocab_size() const;
    
    /**
     * Get token index from string
     */
    int token_to_idx(const std::string& token) const;
    
    /**
     * Get token string from index
     */
    std::string idx_to_token(int idx) const;

private:
    DecoderConfig config_;
    
    // Decoder components
    std::unique_ptr<fl::lib::text::LexiconDecoder> lexicon_decoder_;
    std::unique_ptr<fl::lib::text::Dictionary> tokens_dict_;
    std::unique_ptr<fl::lib::text::Dictionary> word_dict_;
    std::unique_ptr<fl::lib::text::Trie> trie_;
    std::unique_ptr<lm::ngram::Model> kenlm_model_;
    
    // Token indices
    int blank_idx_;
    int sil_idx_;
    int unk_idx_;
    
    // Token mappings
    std::map<std::string, int> token_to_index_;
    std::map<int, std::string> index_to_token_;
    
    /**
     * Load tokens from file
     */
    bool load_tokens();
    
    /**
     * Load lexicon from file
     */
    bool load_lexicon();
    
    /**
     * Load KenLM model
     */
    bool load_lm();
    
    /**
     * Build trie structure for lexicon-based decoding
     */
    bool build_trie();
    
    /**
     * Apply log softmax to logits
     */
    void log_softmax(const float* logits, float* log_probs, int T, int V);
};

/**
 * Feature Extractor class
 * 
 * Extracts hand shape, hand position, and lip features from landmarks
 */
class FeatureExtractor {
public:
    FeatureExtractor();
    
    /**
     * Extract features from landmarks
     * 
     * @param landmarks Current frame landmarks
     * @param prev_landmarks Previous frame landmarks (for velocity)
     * @param prev2_landmarks Frame t-2 landmarks (for acceleration)
     * @return Extracted features
     */
    FrameFeatures extract(
        const LandmarkResults& landmarks,
        const LandmarkResults* prev_landmarks = nullptr,
        const LandmarkResults* prev2_landmarks = nullptr
    );

private:
    // Feature extraction helper functions
    float scalar_distance(float x1, float y1, float z1, 
                         float x2, float y2, float z2);
    
    float polygon_area(const std::vector<float>& xs, 
                      const std::vector<float>& ys);
    
    float mean_contour_curvature(const std::vector<std::pair<float, float>>& points);
    
    float get_angle(float x1, float y1, float z1,
                   float x2, float y2, float z2,
                   float x3, float y3, float z3);
};

/**
 * Overlap-Save Window Processor
 * 
 * Manages streaming decoding with overlap-save windowing
 */
class WindowProcessor {
public:
    WindowProcessor(CTCDecoder* decoder, TFLiteSequenceModel* sequence_model);
    
    /**
     * Reset the processor for a new stream
     */
    void reset();
    
    /**
     * Push a new frame of features
     * 
     * @param features Frame features
     * @return true if a window is ready to process
     */
    bool push_frame(const FrameFeatures& features);
    
    /**
     * Process current window and get decoded result
     * 
     * @param tflite_model_runner Function to run TFLite model on features
     * @return Recognition result (empty if no update)
     */
    RecognitionResult process_window();
    
    /**
     * Finalize and get last result
     */
    RecognitionResult finalize();

private:
    CTCDecoder* decoder_;
    TFLiteSequenceModel* sequence_model_;
    
    std::deque<FrameFeatures> valid_features_;
    std::vector<std::vector<float>> all_logits_;  // Accumulated committed logits
    
    int chunk_idx_;
    int next_window_needed_;
    int frame_count_;
    int effective_vocab_size_;
    
    /**
     * Process a single window
     */
    std::vector<float> process_single_window(
        int window_start,
        int window_end,
        int commit_start,
        int commit_end,
        int& out_vocab_size
    );
};

/**
 * Homophone-based sentence correction
 * 
 * Uses KenLM to select best word from homophones
 */
class SentenceCorrector {
public:
    /**
     * Constructor
     * 
     * @param homophones_path Path to homophones JSONL file
     * @param kenlm_path Path to KenLM model for French
     */
    SentenceCorrector(const std::string& homophones_path,
                     const std::string& kenlm_path);
    
    /**
     * Initialize the corrector
     */
    bool initialize();
    
    /**
     * Correct a LIAPHON phoneme sequence to French text
     * 
     * @param liaphon_phonemes Vector of LIAPHON phonemes
     * @return Corrected French sentence
     */
    std::string correct(const std::vector<std::string>& liaphon_phonemes);

private:
    std::string homophones_path_;
    std::string kenlm_path_;
    
    std::map<std::string, std::vector<std::string>> ipa_to_homophones_;
    std::unique_ptr<lm::ngram::Model> kenlm_model_;
    
    /**
     * Beam search over homophones
     */
    std::vector<std::string> beam_search(
        const std::vector<std::vector<std::string>>& homophone_lists,
        int beam_width = 20
    );
};

/**
 * Phoneme mappings
 */
extern const std::map<std::string, std::string> IPA_TO_LIAPHON;
extern const std::map<std::string, std::string> LIAPHON_TO_IPA;

/**
 * Convert LIAPHON to IPA
 */
std::string liaphon_to_ipa(const std::vector<std::string>& liaphon);

/**
 * Convert IPA to LIAPHON
 */
std::vector<std::string> ipa_to_liaphon(const std::string& ipa);

bool write_subtitled_video(
    const std::string& input_path,
    const std::deque<RecognitionResult>& recognition_results,
    const std::string& output_path,
    double fps);

} // namespace cued_speech

#endif // CUED_SPEECH_DECODER_H

