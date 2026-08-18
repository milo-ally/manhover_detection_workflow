#include "rknpu_manhole.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

void draw_detections(cv::Mat &frame, const std::vector<Detection> &detections) {
    const cv::Scalar colors[] = {
        {46, 204, 113}, {52, 73, 235}, {0, 165, 255}, {0, 0, 255}, {255, 191, 0}
    };
    for (const auto &det : detections) {
        const cv::Scalar color = colors[det.class_id % 5];
        cv::rectangle(frame, det.box, color, 2);
        std::ostringstream label;
        label << class_name(det.class_id) << " " << std::fixed << std::setprecision(2)
              << det.confidence;
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX,
                                                   0.55, 1, &baseline);
        const int x = std::max(0, det.box.x);
        const int y = std::max(text_size.height + 4, det.box.y);
        cv::rectangle(frame, cv::Rect(x, y - text_size.height - 6,
                                      text_size.width + 6, text_size.height + 6), color, -1);
        cv::putText(frame, label.str(), cv::Point(x + 3, y - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 7) {
        std::cerr << "usage: " << argv[0]
                  << " <model.rknn> <input_video> <output_video>"
                  << " [conf_threshold] [iou_threshold] [max_det]\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];
    const float conf_threshold = argc > 4 ? std::stof(argv[4]) : 0.25f;
    const float iou_threshold = argc > 5 ? std::stof(argv[5]) : 0.45f;
    const int max_det = argc > 6 ? std::stoi(argv[6]) : 100;

    cv::VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "cannot open input video: " << input_path << std::endl;
        return 1;
    }
    const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = capture.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0 && std::isfinite(fps))) {
        fps = 25.0;
    }

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "cannot open output video: " << output_path
                  << "; check OpenCV video codec support" << std::endl;
        return 1;
    }

    ManholeRknn detector;
    if (!detector.init(model_path)) {
        return 1;
    }
    std::cout << "video=" << width << "x" << height << " fps=" << fps
              << " model_input=" << detector.input_width() << "x"
              << detector.input_height() << std::endl;

    cv::Mat frame;
    size_t frame_index = 0;
    double inference_ms_sum = 0.0;
    while (capture.read(frame)) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<Detection> detections;
        if (!detector.infer(frame, detections, conf_threshold, iou_threshold, max_det)) {
            std::cerr << "inference failed at frame " << frame_index << std::endl;
            return 1;
        }
        const auto end = std::chrono::steady_clock::now();
        const double inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
        inference_ms_sum += inference_ms;
        draw_detections(frame, detections);
        writer.write(frame);

        if (frame_index % 30 == 0) {
            std::cout << "frame=" << frame_index << " detections=" << detections.size()
                      << " inference_ms=" << inference_ms << std::endl;
        }
        ++frame_index;
    }
    writer.release();
    capture.release();
    std::cout << "saved " << frame_index << " frames to " << output_path;
    if (frame_index > 0) {
        std::cout << ", average inference_ms=" << inference_ms_sum / frame_index;
    }
    std::cout << std::endl;
    return 0;
}
