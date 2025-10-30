/**
 * Example usage of the Cued Speech Decoder C API
 * 
 * Compile:
 *   gcc -o example example_usage.c -lcued_speech_decoder -lm
 * 
 * Run:
 *   ./example lexicon.txt tokens.txt lm.bin [model.tflite]
 */

#include "decoder_c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <lexicon> <tokens> <lm> [model.tflite]\n", argv[0]);
        return 1;
    }
    
    const char* lexicon_path = argv[1];
    const char* tokens_path = argv[2];
    const char* lm_path = argv[3];
    const char* tflite_path = (argc > 4) ? argv[4] : NULL;
    
    printf("=== Cued Speech Decoder Example ===\n\n");
    
    // =====================================================================
    // 1. Create and Initialize Decoder
    // =====================================================================
    
    printf("1. Initializing decoder...\n");
    
    DecoderConfig config = decoder_config_default();
    config.lexicon_path = lexicon_path;
    config.tokens_path = tokens_path;
    config.lm_path = lm_path;
    config.lm_dict_path = NULL;
    config.nbest = 1;
    config.beam_size = 40;
    config.beam_threshold = 50.0f;
    config.lm_weight = 3.23f;
    config.word_score = 0.0f;
    config.sil_score = 0.0f;
    
    DecoderHandle decoder = decoder_create(&config);
    
    if (!decoder) {
        fprintf(stderr, "Failed to create decoder: %s\n", decoder_get_last_error());
        return 1;
    }
    
    int vocab_size = decoder_get_vocab_size(decoder);
    printf("   Decoder initialized! Vocabulary size: %d\n\n", vocab_size);
    
    // =====================================================================
    // 2. Single-Shot Decoding Example
    // =====================================================================
    
    printf("2. Single-shot decoding example...\n");
    
    // Create mock logits [T x V]
    int T = 50;  // time steps
    float* logits = (float*)malloc(T * vocab_size * sizeof(float));
    
    // Fill with random logits (in practice, these come from your CTC model)
    for (int t = 0; t < T; t++) {
        for (int v = 0; v < vocab_size; v++) {
            logits[t * vocab_size + v] = ((float)rand() / RAND_MAX) - 0.5f;
        }
    }
    
    // Decode
    int num_results = 0;
    Hypothesis* results = decoder_decode(decoder, logits, T, vocab_size, &num_results);
    
    if (results && num_results > 0) {
        printf("   Decoded %d hypotheses:\n", num_results);
        printf("   Best hypothesis (score: %.3f):\n", results[0].score);
        printf("     Words: ");
        for (int i = 0; i < results[0].words_length; i++) {
            printf("%s ", results[0].words[i]);
        }
        printf("\n");
        printf("     Tokens: ");
        for (int i = 0; i < results[0].tokens_length; i++) {
            const char* token = decoder_idx_to_token(decoder, results[0].tokens[i]);
            printf("%s ", token ? token : "?");
        }
        printf("\n\n");
        
        decoder_free_hypotheses(results, num_results);
    } else {
        printf("   No results (this is expected with random logits)\n\n");
    }
    
    free(logits);
    
    // =====================================================================
    // 3. Streaming Decoding Example
    // =====================================================================
    
    printf("3. Streaming decoding example...\n");
    
    StreamHandle stream = stream_create(decoder);
    if (!stream) {
        fprintf(stderr, "Failed to create stream: %s\n", decoder_get_last_error());
        decoder_destroy(decoder);
        return 1;
    }
    
    if (tflite_path) {
        if (!stream_load_tflite_model(stream, tflite_path)) {
            fprintf(stderr, "Failed to load TFLite model '%s': %s\n", tflite_path, decoder_get_last_error());
        }
    } else {
        printf("   No TFLite model provided. Streaming outputs will be empty.\n");
    }
    
    printf("   Stream created. Pushing frames...\n");
    
    // Simulate pushing frames with mock features
    int num_frames = 150;  // Push enough to trigger window processing
    int frames_pushed = 0;
    
    for (int i = 0; i < num_frames; i++) {
        // Create mock features [33 floats: 7 hand_shape + 18 hand_pos + 8 lips]
        float features[33];
        for (int j = 0; j < 33; j++) {
            features[j] = ((float)rand() / RAND_MAX);
        }
        
        bool window_ready = stream_push_frame(stream, features);
        frames_pushed++;
        
        if (window_ready) {
            printf("   Window ready at frame %d. Processing...\n", i);
            
            RecognitionResult* result = stream_process_window(stream);
            
            if (result && result->phonemes_length > 0) {
                printf("   Result at frame %d:\n", result->frame_number);
                printf("     Phonemes: ");
                for (int j = 0; j < result->phonemes_length; j++) {
                    printf("%s ", result->phonemes[j]);
                }
                printf("\n");
                printf("     Confidence: %.3f\n", result->confidence);
                
                if (result->french_sentence) {
                    printf("     French: %s\n", result->french_sentence);
                }
            } else {
                printf("   (No result produced — ensure TFLite model is loaded.)\n");
            }
            
            stream_free_result(result);
        }
    }
    
    printf("   Pushed %d frames total.\n", frames_pushed);
    
    printf("   Finalizing stream...\n");
    RecognitionResult* final_result = stream_finalize(stream);
    
    if (final_result && final_result->phonemes_length > 0) {
        printf("   Final result:\n");
        printf("     Phonemes: ");
        for (int j = 0; j < final_result->phonemes_length; j++) {
            printf("%s ", final_result->phonemes[j]);
        }
        printf("\n");
    } else {
        printf("   (No final result produced.)\n");
    }
    
    stream_free_result(final_result);
    stream_destroy(stream);
    
    printf("\n");
    
    // =====================================================================
    // 4. Phoneme Conversion Example
    // =====================================================================
    
    printf("4. Phoneme conversion example...\n");
    
    const char* liaphon_phonemes[] = {"b", "o~", "z^", "u", "r"};
    int num_phonemes = 5;
    
    char* ipa_result = phoneme_liaphon_to_ipa(liaphon_phonemes, num_phonemes);
    if (ipa_result) {
        printf("   LIAPHON: ");
        for (int i = 0; i < num_phonemes; i++) {
            printf("%s ", liaphon_phonemes[i]);
        }
        printf("\n");
        printf("   IPA: %s\n\n", ipa_result);
        corrector_free_string(ipa_result);
    }
    
    // =====================================================================
    // 5. Clean Up
    // =====================================================================
    
    printf("5. Cleaning up...\n");
    decoder_destroy(decoder);
    printf("   Done!\n\n");
    
    return 0;
}

