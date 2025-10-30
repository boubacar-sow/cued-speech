#include "decoder.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using cued_speech::DecoderConfig;
using cued_speech::CTCDecoder;
using cued_speech::FrameFeatures;
using cued_speech::TFLiteSequenceModel;
using cued_speech::WindowProcessor;
using cued_speech::SentenceCorrector;
using cued_speech::RecognitionResult;

namespace {

const char* kFeatureScript = R"PY(
import argparse
import cv2
import sys
from collections import deque
from cued_speech.decoder_tflite import MediaPipeStyleLandmarkExtractor, extract_features_single_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--face', required=True)
    parser.add_argument('--hand', required=True)
    parser.add_argument('--pose', required=True)
    args = parser.parse_args()

    extractor = MediaPipeStyleLandmarkExtractor(
        face_model_path=args.face if args.face else None,
        hand_model_path=args.hand if args.hand else None,
        pose_model_path=args.pose if args.pose else None,
    )

    cap = cv2.VideoCapture(args.video)
    coordinate_buffer = deque(maxlen=3)
    hs_keys = None
    hp_keys = None
    lp_keys = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = extractor.process(rgb_frame)

            landmarks_data = {}

            if results.right_hand_landmarks:
                for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                    landmarks_data[f"hand_x{i}"] = float(landmark.x)
                    landmarks_data[f"hand_y{i}"] = float(landmark.y)
                    landmarks_data[f"hand_z{i}"] = float(landmark.z)

            if results.face_landmarks:
                for i, landmark in enumerate(results.face_landmarks.landmark):
                    landmarks_data[f"face_x{i}"] = float(landmark.x)
                    landmarks_data[f"face_y{i}"] = float(landmark.y)
                    landmarks_data[f"face_z{i}"] = float(landmark.z)
                    landmarks_data[f"lip_x{i}"] = float(landmark.x)
                    landmarks_data[f"lip_y{i}"] = float(landmark.y)
                    landmarks_data[f"lip_z{i}"] = float(landmark.z)

            row = dict(landmarks_data)
            coordinate_buffer.append(row)
            prev = coordinate_buffer[-2] if len(coordinate_buffer) >= 2 else None
            prev2 = coordinate_buffer[-3] if len(coordinate_buffer) >= 3 else None
            features = extract_features_single_row(row, prev, prev2)

            if features:
                hs_candidates = [k for k in features.keys() if 'hand' in k and 'face' not in k]
                hp_candidates = [k for k in features.keys() if 'face' in k]
                lp_candidates = [k for k in features.keys() if 'lip' in k]

                if len(hs_candidates) == 7 and len(hp_candidates) == 18 and len(lp_candidates) == 8:
                    if hs_keys is None:
                        hs_keys = hs_candidates
                        hp_keys = hp_candidates
                        lp_keys = lp_candidates

                    values = [features[k] for k in hs_keys]
                    values.extend(features[k] for k in hp_keys)
                    values.extend(features[k] for k in lp_keys)

                    line = "DATA,{},{}".format(
                        frame_idx,
                        ",".join(f"{v:.10f}" for v in values),
                    )
                    sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                    continue

            sys.stdout.write(f"DROP,{frame_idx}\n")
            sys.stdout.flush()
    finally:
        cap.release()
        extractor.close()


if __name__ == '__main__':
    main()
)PY";

bool ensure_feature_script(const fs::path& script_path) {
    std::ofstream script(script_path, std::ios::trunc);
    if (!script) {
        std::cerr << "Failed to create helper script at " << script_path << std::endl;
        return false;
    }
    script << kFeatureScript;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path repo_root = fs::path("/store/scratch/bsow/Documents/cued_speech");
        fs::path download_dir = repo_root / "download";
        fs::path output_dir = repo_root / "output" / "cpp_demo";
        fs::create_directories(output_dir);

        fs::path video_path = download_dir / "test_decode_mjpg.avi";
        fs::path model_path = download_dir / "cuedspeech_model_fixed_temporal.tflite";
        fs::path tokens_path = download_dir / "phonelist.csv";
        fs::path lexicon_path = download_dir / "lexicon.txt";
        fs::path kenlm_fr_path = download_dir / "kenlm_fr.bin";
        fs::path kenlm_ipa_path = download_dir / "kenlm_ipa.binary";
        fs::path homophones_path = download_dir / "homophones_dico.jsonl";
        fs::path face_model_path = download_dir / "face_landmarker.task";
        fs::path hand_model_path = download_dir / "hand_landmarker.task";
        fs::path pose_model_path = download_dir / "pose_landmarker_full.task";
        fs::path script_path = download_dir / "generate_features.py";
        fs::path python_exe = fs::path("python");

        if (!fs::exists(video_path)) {
            std::cerr << "Input video not found: " << video_path << std::endl;
            return 1;
        }

        if (!fs::exists(model_path)) {
            std::cerr << "Acoustic TFLite model not found at " << model_path << std::endl;
            return 1;
        }

        if (!ensure_feature_script(script_path)) {
            return 1;
        }

        DecoderConfig config;
        config.lexicon_path = lexicon_path.string();
        config.tokens_path = tokens_path.string();
        config.lm_path = kenlm_ipa_path.string();
        config.nbest = 1;
        config.beam_size = 40;
        config.beam_threshold = 50.0f;
        config.lm_weight = 3.23f;
        config.word_score = 0.0f;
        config.sil_score = 0.0f;

        CTCDecoder decoder(config);
        if (!decoder.initialize()) {
            std::cerr << "Failed to initialize CTC decoder." << std::endl;
            return 1;
        }

        TFLiteSequenceModel acoustic_model;
        if (!acoustic_model.load(model_path.string())) {
            std::cerr << "Failed to load acoustic TFLite model: " << model_path << std::endl;
            return 1;
        }

        WindowProcessor processor(&decoder, &acoustic_model);

        std::stringstream cmd;
        cmd << '"' << python_exe.string() << '"'
            << ' ' << '"' << script_path.string() << '"'
            << " --video " << '"' << video_path.string() << '"'
            << " --face " << '"' << face_model_path.string() << '"'
            << " --hand " << '"' << hand_model_path.string() << '"'
            << " --pose " << '"' << pose_model_path.string() << '"';

        std::cout << "Streaming frame features via Python..." << std::endl;
        FILE* pipe = popen(cmd.str().c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to launch feature extraction helper." << std::endl;
            return 1;
        }

        std::vector<RecognitionResult> recognitions;
        recognitions.reserve(16);

        int total_frames = 0;
        int valid_frames = 0;
        int dropped_frames = 0;
        int last_frame_number = 0;

        std::array<char, 8192> buffer{};
        std::string pending_line;

        auto process_line = [&](const std::string& raw_line) {
            if (raw_line.empty()) {
                return;
            }

            if (raw_line.rfind("DATA,", 0) == 0) {
                total_frames++;
                valid_frames++;

                std::string payload = raw_line.substr(5);
                std::stringstream ss(payload);
                std::string token;
                if (!std::getline(ss, token, ',')) {
                    return;
                }

                int frame_number = 0;
                try {
                    frame_number = std::stoi(token);
                } catch (const std::exception&) {
                    return;
                }
                last_frame_number = frame_number;

                std::vector<float> values;
                values.reserve(33);
                while (std::getline(ss, token, ',')) {
                    try {
                        values.push_back(std::stof(token));
                    } catch (const std::exception&) {
                        values.push_back(0.0f);
                    }
                }

                if (values.size() != 33) {
                    std::cerr << "Warning: expected 33 feature values, received "
                              << values.size() << " for frame " << frame_number << std::endl;
                    return;
                }

                FrameFeatures feats;
                feats.hand_shape.assign(values.begin(), values.begin() + 7);
                feats.hand_position.assign(values.begin() + 7, values.begin() + 25);
                feats.lips.assign(values.begin() + 25, values.end());

                bool ready = processor.push_frame(feats);
                if (ready) {
                    auto partial = processor.process_window();
                    if (!partial.phonemes.empty()) {
                        partial.frame_number = frame_number;
                        if (!recognitions.empty()) {
                            recognitions.clear();
                        }
                        recognitions.push_back(std::move(partial));
                    }
                }
            } else if (raw_line.rfind("DROP,", 0) == 0) {
                total_frames++;
                dropped_frames++;
            }
        };

        while (true) {
            char* chunk = std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe);
            if (!chunk) {
                if (!pending_line.empty()) {
                    process_line(pending_line);
                    pending_line.clear();
                }
                break;
            }

            pending_line.append(chunk);
            if (!pending_line.empty() && pending_line.back() == '\n') {
                while (!pending_line.empty() && (pending_line.back() == '\n' || pending_line.back() == '\r')) {
                    pending_line.pop_back();
                }
                process_line(pending_line);
                pending_line.clear();
            }
        }

        int script_status = pclose(pipe);
        if (script_status != 0) {
            std::cerr << "Feature extraction helper exited with status " << script_status << std::endl;
        }

        auto final_partial = processor.finalize();
        if (!final_partial.phonemes.empty()) {
            final_partial.frame_number = last_frame_number;
            if (!recognitions.empty()) {
                recognitions.clear();
            }
            recognitions.push_back(std::move(final_partial));
        }

        SentenceCorrector corrector(homophones_path.string(), kenlm_fr_path.string());
        if (corrector.initialize()) {
            for (auto& res : recognitions) {
                std::string corrected = corrector.correct(res.phonemes);
                if (!corrected.empty()) {
                    res.french_sentence = std::move(corrected);
                }
            }
        } else {
            std::cerr << "Warning: failed to initialize sentence corrector. Subtitles will show phonemes only." << std::endl;
        }

        std::cout << "\nTotal valid frames: " << valid_frames
                  << " (out of " << total_frames << " total frames)";
        if (dropped_frames > 0) {
            std::cout << " -- dropped " << dropped_frames << " frames due to incomplete landmarks";
        }
        std::cout << std::endl;
        std::cout << "Total chunks processed: " << processor.chunks_processed() << std::endl;

        if (!recognitions.empty()) {
            const auto& final_result = recognitions.back();
            std::cout << "\nFinal phoneme sequence: ";
            for (size_t i = 0; i < final_result.phonemes.size(); ++i) {
                if (i > 0) {
                    std::cout << ' ';
                }
                std::cout << final_result.phonemes[i];
            }
            std::cout << std::endl;
            if (!final_result.french_sentence.empty()) {
                std::cout << "French sentence: " << final_result.french_sentence << std::endl;
            }
        } else {
            std::cout << "No decoded phoneme sequence available." << std::endl;
        }

        std::deque<RecognitionResult> recognition_deque(recognitions.begin(), recognitions.end());
        fs::path output_video = output_dir / "decoded_cpp.avi";
        if (!write_subtitled_video(video_path.string(), recognition_deque, output_video.string(), 0.0)) {
            std::cerr << "Failed to write subtitled video." << std::endl;
            return 1;
        }

        std::cout << "✅ Decoding complete. Output saved to " << output_video << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
