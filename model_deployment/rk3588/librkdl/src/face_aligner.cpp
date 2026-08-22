#include "face_aligner.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#ifdef HAVE_DLIB
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#endif

namespace rkdl {

struct FaceAligner::Impl {
    bool ready = false;
#ifdef HAVE_DLIB
    dlib::shape_predictor predictor;
#endif
};

FaceAligner::FaceAligner() : impl_(new Impl) {}
FaceAligner::~FaceAligner() = default;

bool FaceAligner::load(const std::string& predictor_path) {
#ifdef HAVE_DLIB
    try {
        dlib::deserialize(predictor_path) >> impl_->predictor;
        impl_->ready = true;
    } catch (...) { impl_->ready = false; }
#else
    (void)predictor_path;
#endif
    return impl_->ready;
}

bool FaceAligner::align112Rgb(const VideoFrame& face, cv::Mat& aligned_rgb) const {
#ifndef HAVE_DLIB
    (void)face; (void)aligned_rgb;
    return false;
#else
    if (!impl_->ready || !face.data || face.width < 8 || face.height < 8) return false;
    cv::Mat bgr;
    if (face.format == PixelFormat::BGR888) {
        bgr = cv::Mat(face.height, face.width, CV_8UC3, const_cast<std::uint8_t*>(face.data), face.stride);
    } else if (face.format == PixelFormat::RGB888) {
        cv::Mat rgb(face.height, face.width, CV_8UC3, const_cast<std::uint8_t*>(face.data), face.stride);
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    } else return false;
    try {
        dlib::cv_image<dlib::bgr_pixel> image(bgr);
        const dlib::rectangle rectangle(0, 0, face.width - 1, face.height - 1);
        const dlib::full_object_detection shape = impl_->predictor(image, rectangle);
        if (shape.num_parts() < 68) return false;
        auto average = [&](int a, int b) {
            return cv::Point2f((shape.part(a).x() + shape.part(b).x()) * 0.5f,
                               (shape.part(a).y() + shape.part(b).y()) * 0.5f);
        };
        const std::vector<cv::Point2f> source = {
            average(36, 39), average(42, 45),
            cv::Point2f(shape.part(30).x(), shape.part(30).y()),
            cv::Point2f(shape.part(48).x(), shape.part(48).y()),
            cv::Point2f(shape.part(54).x(), shape.part(54).y())};
        const std::vector<cv::Point2f> target = {
            {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
            {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
        const cv::Mat transform = cv::estimateAffinePartial2D(source, target);
        if (transform.empty()) return false;
        cv::Mat aligned_bgr;
        cv::warpAffine(bgr, aligned_bgr, transform, cv::Size(112, 112), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::cvtColor(aligned_bgr, aligned_rgb, cv::COLOR_BGR2RGB);
        return true;
    } catch (...) { return false; }
#endif
}

bool FaceAligner::available() const { return impl_->ready; }

}  // namespace rkdl
