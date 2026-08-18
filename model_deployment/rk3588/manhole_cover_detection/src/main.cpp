#include "rknpu_manhole.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdio>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

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

bool is_rtsp_url(const std::string &value) {
    return value.rfind("rtsp://", 0) == 0 || value.rfind("rtsps://", 0) == 0;
}

std::string shell_quote(const std::string &value) {
    std::string quoted = "'";
    for (char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

}  // namespace

int main(int argc, char **argv) {
    std::string model_path;
    std::string input_path;
    std::string output_path;
    float conf_threshold = 0.25f;
    float iou_threshold = 0.45f;
    int max_det = 100;

    auto next_value = [&](int &index, const char *option) -> const char * {
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string("missing value for ") + option);
        }
        return argv[++index];
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--model") {
                model_path = next_value(i, "--model");
            } else if (option == "--input") {
                input_path = next_value(i, "--input");
            } else if (option == "--output") {
                output_path = next_value(i, "--output");
            } else if (option == "--conf-thres") {
                conf_threshold = std::stof(next_value(i, "--conf-thres"));
            } else if (option == "--iou-thres") {
                iou_threshold = std::stof(next_value(i, "--iou-thres"));
            } else if (option == "--max-det") {
                max_det = std::stoi(next_value(i, "--max-det"));
            } else if (option == "--help" || option == "-h") {
                std::cout << "usage: " << argv[0]
                          << " --model model.rknn --input input.mp4 --output output.mp4"
                          << " [--conf-thres 0.25] [--iou-thres 0.45] [--max-det 100]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "argument error: " << error.what() << std::endl;
        return 2;
    }

    if (model_path.empty() || input_path.empty() || output_path.empty() ||
        conf_threshold < 0.0f || iou_threshold < 0.0f || max_det <= 0) {
        std::cerr << "usage: " << argv[0]
                  << " --model model.rknn --input input.mp4 --output output.mp4"
                  << " [--conf-thres 0.25] [--iou-thres 0.45] [--max-det 100]\n";
        return 2;
    }

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

    cv::VideoWriter writer;
    FILE *stream_pipe = nullptr;
    if (is_rtsp_url(output_path)) {
        std::ostringstream command;
        command << "ffmpeg -loglevel warning -f rawvideo -pix_fmt bgr24"
                << " -s " << width << "x" << height
                << " -r " << fps << " -i pipe:0 -an"
                << " -c:v h264_rkmpp -pix_fmt yuv420p -f rtsp"
                << " -rtsp_transport tcp " << shell_quote(output_path);
        std::cout << "output_mode=rtsp command=" << command.str() << std::endl;
        stream_pipe = popen(command.str().c_str(), "w");
        if (!stream_pipe) {
            std::cerr << "cannot start ffmpeg RTSP output: " << output_path << std::endl;
            return 1;
        }
    } else {
        writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "cannot open output video: " << output_path
                      << "; check OpenCV video codec support" << std::endl;
            return 1;
        }
        std::cout << "output_mode=file" << std::endl;
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
        if (stream_pipe) {
            if (!frame.isContinuous()) {
                frame = frame.clone();
            }
            const size_t bytes = frame.total() * frame.elemSize();
            if (fwrite(frame.data, 1, bytes, stream_pipe) != bytes) {
                std::cerr << "ffmpeg RTSP output pipe closed at frame " << frame_index << std::endl;
                pclose(stream_pipe);
                return 1;
            }
            fflush(stream_pipe);
        } else {
            writer.write(frame);
        }

        std::cout << "frame=" << frame_index << " detections=" << detections.size()
                  << " inference_ms=" << inference_ms << std::endl;
        ++frame_index;
    }
    if (stream_pipe) {
        pclose(stream_pipe);
    } else {
        writer.release();
    }
    capture.release();
    std::cout << "saved " << frame_index << " frames to " << output_path;
    if (frame_index > 0) {
        std::cout << ", average inference_ms=" << inference_ms_sum / frame_index;
    }
    std::cout << std::endl;
    return 0;
}
