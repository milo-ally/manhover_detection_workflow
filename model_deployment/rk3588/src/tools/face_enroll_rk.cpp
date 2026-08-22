#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include "rknn_model.h"
#include "face_aligner.h"

namespace {
struct Args {
    std::string image, name, detector, arcface, landmark, database;
};

bool parse(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc) return false;
        const std::string key = argv[i], value = argv[++i];
        if (key == "--image") args.image = value;
        else if (key == "--name") args.name = value;
        else if (key == "--face-det-model") args.detector = value;
        else if (key == "--arcface-model") args.arcface = value;
        else if (key == "--landmark-dat") args.landmark = value;  // retained for CLI compatibility
        else if (key == "--db-txt") args.database = value;
        else return false;
    }
    return !args.image.empty() && !args.name.empty() && !args.detector.empty() &&
           !args.arcface.empty() && !args.landmark.empty() && !args.database.empty();
}

VideoFrame view(const cv::Mat& image) {
    VideoFrame frame;
    frame.width = image.cols; frame.height = image.rows;
    frame.stride = static_cast<int>(image.step); frame.format = PixelFormat::BGR888;
    frame.data = image.data; frame.size = image.total() * image.elemSize();
    return frame;
}

bool update_database(const std::string& path, const std::string& name, const std::vector<float>& embedding) {
    std::vector<std::string> lines;
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line); std::string existing;
        if (row >> existing && existing != name) lines.push_back(line);
    }
    std::ostringstream next; next << name;
    for (float value : embedding) next << ' ' << value;
    lines.push_back(next.str());
    std::ofstream output(path, std::ios::trunc);
    if (!output) return false;
    for (const auto& value : lines) output << value << '\n';
    return true;
}
}

int main(int argc, char** argv) {
    Args args;
    if (!parse(argc, argv, args)) {
        std::cerr << "Usage: face_enroll --image <jpg> --name <id> --face-det-model <rknn> "
                     "--arcface-model <rknn> --landmark-dat <dat> --db-txt <txt>\n";
        return 2;
    }
    cv::Mat image = cv::imread(args.image, cv::IMREAD_COLOR);
    if (image.empty()) { std::cerr << "failed to read image\n"; return 3; }
    rkdl::RknnModel detector("face_detection");
    if (detector.Init(args.detector.c_str()) != 0) { std::cerr << detector.lastError() << '\n'; return 4; }
    AI_RESULT_T detected{};
    VideoFrame image_frame = view(image);
    if (detector.Inference(&image_frame, &detected) != 0 || detected.nObjSize == 0) {
        std::cerr << "no face detected\n"; return 5;
    }
    const AI_OBJ_T* largest = &detected.objects[0];
    for (std::uint32_t i = 1; i < detected.nObjSize; ++i)
        if (detected.objects[i].w * detected.objects[i].h > largest->w * largest->h) largest = &detected.objects[i];
    cv::Rect area(static_cast<int>(largest->x * image.cols), static_cast<int>(largest->y * image.rows),
                  static_cast<int>(largest->w * image.cols), static_cast<int>(largest->h * image.rows));
    area &= cv::Rect(0, 0, image.cols, image.rows);
    if (area.empty()) return 5;
    cv::Mat face = image(area).clone();
    rkdl::FaceAligner aligner;
    cv::Mat aligned_rgb;
    VideoFrame unaligned = view(face);
    const bool aligned = aligner.load(args.landmark) && aligner.align112Rgb(unaligned, aligned_rgb);
    rkdl::RknnModel arcface("face_recognition");
    if (arcface.Init(args.arcface.c_str()) != 0) { std::cerr << arcface.lastError() << '\n'; return 6; }
    std::vector<rkdl::TensorView> outputs;
    VideoFrame face_frame = aligned ? view(aligned_rgb) : view(face);
    if (aligned) face_frame.format = PixelFormat::RGB888;
    if (arcface.RunRaw(&face_frame, &outputs) != 0 || outputs.empty() || outputs[0].values.empty()) {
        std::cerr << "ArcFace inference failed\n"; return 7;
    }
    std::vector<float> embedding = outputs[0].values;
    double norm = 0.0;
    for (float value : embedding) norm += static_cast<double>(value) * value;
    if (norm <= 1e-12) return 7;
    const float inverse = 1.0f / static_cast<float>(std::sqrt(norm));
    for (float& value : embedding) value *= inverse;
    if (!update_database(args.database, args.name, embedding)) { std::cerr << "failed to update database\n"; return 8; }
    std::cout << "enrolled " << args.name << " (" << embedding.size() << " dimensions)\n";
    return 0;
}
