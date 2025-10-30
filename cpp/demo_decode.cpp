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
import csv
import cv2
import pandas as pd
from collections import deque
from cued_speech.decoder_tflite import MediaPipeStyleLandmarkExtractor, extract_features_single_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--face', required=True)
    parser.add_argument('--hand', required=True)
    parser.add_argument('--pose', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    extractor = MediaPipeStyleLandmarkExtractor(
        face_model_path=args.face if args.face else None,
        hand_model_path=args.hand if args.hand else None,
        pose_model_path=args.pose if args.pose else None,
    )

    cap = cv2.VideoCapture(args.video)
    coordinate_buffer = deque(maxlen=3)
    valid_features = []
    frame_numbers = []
    frame_idx = 0

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

        row = pd.Series(landmarks_data)
        coordinate_buffer.append(row)
        prev = coordinate_buffer[-2] if len(coordinate_buffer) >= 2 else None
        prev2 = coordinate_buffer[-3] if len(coordinate_buffer) >= 3 else None
        features = extract_features_single_row(row, prev, prev2)

        if features:
            hs_count = sum(1 for k in features.keys() if 'hand' in k and 'face' not in k)
            hp_count = sum(1 for k in features.keys() if 'face' in k)
            lp_count = sum(1 for k in features.keys() if 'lip' in k)
            if hs_count == 7 and hp_count == 18 and lp_count == 8:
                valid_features.append(features)
                frame_numbers.append(frame_idx)

    cap.release()
    extractor.close()

    if not valid_features:
        with open(args.output, 'w', newline='') as sink:
            writer = csv.writer(sink)
            writer.writerow(['frame'])
        return

    df = pd.DataFrame(valid_features)
    hs_cols = [c for c in df.columns if 'hand' in c and 'face' not in c]
    hp_cols = [c for c in df.columns if 'face' in c]
    lp_cols = [c for c in df.columns if 'lip' in c]

    with open(args.output, 'w', newline='') as sink:
        writer = csv.writer(sink)
        writer.writerow(['frame'] + hs_cols + hp_cols + lp_cols)
        for idx in range(len(df)):
            row_vals = [frame_numbers[idx]]
            row_vals += list(df.iloc[idx][hs_cols])
            row_vals += list(df.iloc[idx][hp_cols])
            row_vals += list(df.iloc[idx][lp_cols])
            writer.writerow(row_vals)


if __name__ == '__main__':
    main()
)PY";

bool ensure_feature_script(const fs::path& script_path) {
    if (fs::exists(script_path)) {
        return true;
    }
    std::ofstream script(script_path);
    if (!script) {
        std::cerr << "Failed to create helper script at " << script_path << std::endl;
        return false;
    }
    script << kFeatureScript;
    return true;
}

bool generate_features(const fs::path& python_exe,
                       const fs::path& script_path,
                       const fs::path& video,
                       const fs::path& face,
                       const fs::path& hand,
                       const fs::path& pose,
                       const fs::path& output_csv) {
    if (!ensure_feature_script(script_path)) {
        return false;
    }

    std::stringstream cmd;
    cmd << '"' << python_exe.string() << '"'
        << ' ' << '"' << script_path.string() << '"'
        << " --video " << '"' << video.string() << '"'
        << " --face " << '"' << face.string() << '"'
        << " --hand " << '"' << hand.string() << '"'
        << " --pose " << '"' << pose.string() << '"'
        << " --output " << '"' << output_csv.string() << '"';

    std::cout << "Generating frame features via Python...\n";
    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        std::cerr << "Python feature extraction failed (exit code " << ret << ")\n";
        return false;
    }
    return true;
}

bool load_feature_csv(const fs::path& csv_path,
                      std::vector<int>& frame_numbers,
                      std::vector<std::array<float, 33>>& feature_rows) {
    std::ifstream in(csv_path);
    if (!in) {
        std::cerr << "Failed to open features CSV: " << csv_path << std::endl;
        return false;
    }

    std::string header;
    if (!std::getline(in, header)) {
        std::cerr << "Features CSV is empty: " << csv_path << std::endl;
        return false;
    }

    const size_t expected_columns = 1 + 33;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            cells.push_back(cell);
        }
        if (cells.size() != expected_columns) {
            std::cerr << "Unexpected column count in features row: " << cells.size()
                      << " (expected " << expected_columns << ")\n";
            return false;
        }
        frame_numbers.push_back(std::stoi(cells[0]));
        std::array<float, 33> feats{};
        for (size_t i = 0; i < 33; ++i) {
            feats[i] = std::stof(cells[i + 1]);
        }
        feature_rows.push_back(feats);
    }

    if (feature_rows.empty()) {
        std::cerr << "No features found in CSV." << std::endl;
        return false;
    }

    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        fs::path repo_root = fs::path("/store/scratch/bsow/Documents/cued_speech");
        fs::path download_dir = repo_root / "download";
        fs::path output_dir = repo_root / "output" / "cpp_demo";
        fs::create_directories(output_dir);

        fs::path video_path = download_dir / "test_decode.mp4";
        fs::path model_path = download_dir / "cuedspeech_model_fixed_temporal.tflite";
        fs::path tokens_path = download_dir / "phonelist.csv";
        fs::path lexicon_path = download_dir / "lexicon.txt";
        fs::path kenlm_fr_path = download_dir / "kenlm_fr.bin";
        fs::path homophones_path = download_dir / "homophones_dico.jsonl";
        fs::path face_model_path = download_dir / "face_landmarker.task";
        fs::path hand_model_path = download_dir / "hand_landmarker.task";
        fs::path pose_model_path = download_dir / "pose_landmarker_full.task";
        fs::path features_csv = download_dir / "test_decode_features.csv";
        fs::path script_path = download_dir / "generate_features.py";
        fs::path python_exe = fs::path("python");

        if (!fs::exists(features_csv)) {
            if (!generate_features(python_exe, script_path, video_path,
                                   face_model_path, hand_model_path, pose_model_path,
                                   features_csv)) {
                return 1;
            }
        } else {
            std::cout << "Using cached features: " << features_csv << '\n';
        }

        std::vector<int> frame_numbers;
        std::vector<std::array<float, 33>> feature_rows;
        if (!load_feature_csv(features_csv, frame_numbers, feature_rows)) {
            return 1;
        }

        if (!fs::exists(model_path)) {
            std::cerr << "Acoustic TFLite model not found at " << model_path << std::endl;
            return 1;
        }

        DecoderConfig config;
        config.lexicon_path = lexicon_path.string();
        config.tokens_path = tokens_path.string();
        config.lm_path = kenlm_fr_path.string();
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

        std::vector<RecognitionResult> recognitions;
        recognitions.reserve(feature_rows.size());

        for (size_t i = 0; i < feature_rows.size(); ++i) {
            const auto& row = feature_rows[i];
            FrameFeatures feats;
            feats.hand_shape.assign(row.begin(), row.begin() + 7);
            feats.hand_position.assign(row.begin() + 7, row.begin() + 25);
            feats.lips.assign(row.begin() + 25, row.end());

            bool ready = processor.push_frame(feats);
            if (ready) {
                auto partial = processor.process_window();
                if (!partial.phonemes.empty()) {
                    size_t idx = std::min(i, frame_numbers.size() - 1);
                    partial.frame_number = frame_numbers[idx];
                    recognitions.push_back(std::move(partial));
                }
            }
        }

        auto final_partial = processor.finalize();
        if (!final_partial.phonemes.empty()) {
            final_partial.frame_number = frame_numbers.back();
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

        std::deque<RecognitionResult> recognition_deque(recognitions.begin(), recognitions.end());
        fs::path output_video = output_dir / "decoded_cpp.mp4";
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
