#include "rknn_model.h"
#include "yolo_postprocess.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>
#include <opencv2/imgproc.hpp>

#ifdef HAVE_RKNN
#include <rknn_api.h>
#endif
#ifdef HAVE_RGA
#include <im2d.h>
#endif

namespace rkdl {

struct RknnModel::Impl {
    explicit Impl(std::string value) : profile(std::move(value)) {}
    std::string profile;
    std::string error;
    std::string path;
    int input_width = 640;
    int input_height = 640;
    int input_channels = 3;
    float confidence = 0.45f;
    float nms = 0.45f;
    bool initialized = false;
    std::mutex mutex;
#ifdef HAVE_RKNN
    rknn_context context = 0;
    rknn_input_output_num io_count{};
    rknn_tensor_attr input_attr{};
    std::vector<rknn_tensor_attr> output_attrs;
#endif
};

RknnModel::RknnModel(std::string profile) : impl_(new Impl(std::move(profile))) {}
RknnModel::~RknnModel() { Deinit(); }

int RknnModel::Init(const char* model_path) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!model_path || !*model_path) { impl_->error = "empty RKNN model path"; return -1; }
#ifndef HAVE_RKNN
    impl_->error = "this binary was built without librknnrt; rebuild on RK3588 or set RKNN_SDK_ROOT";
    return -1;
#else
    if (impl_->initialized) {
        rknn_destroy(impl_->context);
        impl_->initialized = false;
    }
    std::ifstream stream(model_path, std::ios::binary);
    if (!stream) { impl_->error = std::string("cannot open RKNN model: ") + model_path; return -1; }
    std::vector<unsigned char> model((std::istreambuf_iterator<char>(stream)), {});
    if (model.empty()) { impl_->error = "RKNN model is empty"; return -1; }
    int rc = rknn_init(&impl_->context, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
    if (rc != RKNN_SUCC) { impl_->error = "rknn_init failed: " + std::to_string(rc); return rc; }

    // RK3588 owns three NPU cores. AUTO lets the runtime share cores safely
    // between the application's independent stream/model contexts.
    rknn_set_core_mask(impl_->context, RKNN_NPU_CORE_AUTO);
    rc = rknn_query(impl_->context, RKNN_QUERY_IN_OUT_NUM, &impl_->io_count, sizeof(impl_->io_count));
    if (rc != RKNN_SUCC || impl_->io_count.n_input != 1) {
        impl_->error = "RKNN model must expose exactly one image input";
        rknn_destroy(impl_->context); impl_->context = 0; return -1;
    }
    impl_->input_attr = {};
    impl_->input_attr.index = 0;
    rc = rknn_query(impl_->context, RKNN_QUERY_INPUT_ATTR, &impl_->input_attr, sizeof(impl_->input_attr));
    if (rc != RKNN_SUCC) { impl_->error = "RKNN input query failed"; rknn_destroy(impl_->context); return rc; }
    if (impl_->input_attr.fmt == RKNN_TENSOR_NHWC) {
        impl_->input_height = impl_->input_attr.dims[1];
        impl_->input_width = impl_->input_attr.dims[2];
        impl_->input_channels = impl_->input_attr.dims[3];
    } else {
        impl_->input_channels = impl_->input_attr.dims[1];
        impl_->input_height = impl_->input_attr.dims[2];
        impl_->input_width = impl_->input_attr.dims[3];
    }
    impl_->output_attrs.resize(impl_->io_count.n_output);
    for (uint32_t i = 0; i < impl_->io_count.n_output; ++i) {
        impl_->output_attrs[i] = {};
        impl_->output_attrs[i].index = i;
        rc = rknn_query(impl_->context, RKNN_QUERY_OUTPUT_ATTR, &impl_->output_attrs[i], sizeof(rknn_tensor_attr));
        if (rc != RKNN_SUCC) { impl_->error = "RKNN output query failed"; rknn_destroy(impl_->context); return rc; }
    }
    impl_->path = model_path;
    impl_->initialized = true;
    impl_->error.clear();
    return 0;
#endif
}

void RknnModel::GetInputSize(int* width, int* height) {
    if (width) *width = impl_->input_width;
    if (height) *height = impl_->input_height;
}

[[maybe_unused]] static bool frame_to_rgb(const VideoFrame& frame, cv::Mat& rgb, std::string& error) {
    if (!frame.data || frame.width <= 0 || frame.height <= 0) { error = "invalid input frame"; return false; }
    try {
        if (frame.format == PixelFormat::RGB888) {
            cv::Mat view(frame.height, frame.width, CV_8UC3, const_cast<std::uint8_t*>(frame.data), frame.stride);
            rgb = view.clone();
        } else if (frame.format == PixelFormat::BGR888) {
            cv::Mat view(frame.height, frame.width, CV_8UC3, const_cast<std::uint8_t*>(frame.data), frame.stride);
            cv::cvtColor(view, rgb, cv::COLOR_BGR2RGB);
        } else {
            const int stride = frame.stride > 0 ? frame.stride : frame.width;
            std::vector<std::uint8_t> tight(static_cast<std::size_t>(frame.width) * frame.height * 3 / 2);
            for (int y = 0; y < frame.height; ++y)
                std::copy_n(frame.data + static_cast<std::size_t>(y) * stride, frame.width,
                            tight.data() + static_cast<std::size_t>(y) * frame.width);
            const std::uint8_t* uv = frame.data + static_cast<std::size_t>(stride) * frame.height;
            for (int y = 0; y < frame.height / 2; ++y)
                std::copy_n(uv + static_cast<std::size_t>(y) * stride, frame.width,
                            tight.data() + static_cast<std::size_t>(frame.width) * (frame.height + y));
            cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1, tight.data());
            cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
        }
        return true;
    } catch (const cv::Exception& e) { error = e.what(); return false; }
}

[[maybe_unused]] static void resize_rgb(const cv::Mat& source, cv::Mat& destination, int width, int height) {
    destination.create(height, width, CV_8UC3);
#ifdef HAVE_RGA
    // Virtual-address RGA is the portable path for decoded CPU frames.  When
    // the decoder exposes a DMA-BUF the same runner ABI can be upgraded to
    // wrapbuffer_fd without changing plugins or application code.
    rga_buffer_t src = wrapbuffer_virtualaddr(source.data, source.cols, source.rows,
                                               RK_FORMAT_RGB_888, source.cols, source.rows);
    rga_buffer_t dst = wrapbuffer_virtualaddr(destination.data, width, height,
                                               RK_FORMAT_RGB_888, width, height);
    if (imresize(src, dst) == IM_STATUS_SUCCESS) return;
#endif
    cv::resize(source, destination, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
}

int RknnModel::Inference(const VideoFrame* frame, AI_RESULT_T* result) {
    if (!frame || !result) return -1;
    *result = {};
    std::vector<TensorView> tensors;
    LetterboxTransform transform;
    const int rc = RunRaw(frame, &tensors, &transform);
    if (rc != 0) return rc;
    return decode_yolo(tensors, transform, impl_->confidence, impl_->nms, impl_->profile, result);
}

int RknnModel::RunRaw(const VideoFrame* frame, std::vector<TensorView>* tensors,
                      LetterboxTransform* transform) {
    if (!frame || !tensors) return -1;
    tensors->clear();
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifndef HAVE_RKNN
    (void)transform;
    impl_->error = "RKNN inference requested from a host-validation build";
    return -1;
#else
    if (!impl_->initialized) { impl_->error = "RKNN model is not initialized"; return -1; }
    cv::Mat rgb;
    if (!frame_to_rgb(*frame, rgb, impl_->error)) return -1;
    const float scale = std::min(static_cast<float>(impl_->input_width) / rgb.cols,
                                 static_cast<float>(impl_->input_height) / rgb.rows);
    const int resized_w = std::max(1, static_cast<int>(std::round(rgb.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(rgb.rows * scale)));
    const int pad_x = (impl_->input_width - resized_w) / 2;
    const int pad_y = (impl_->input_height - resized_h) / 2;
    cv::Mat resized, input(impl_->input_height, impl_->input_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resize_rgb(rgb, resized, resized_w, resized_h);
    resized.copyTo(input(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

    rknn_input rknn_in{};
    rknn_in.index = 0;
    rknn_in.type = RKNN_TENSOR_UINT8;
    rknn_in.fmt = RKNN_TENSOR_NHWC;
    rknn_in.size = static_cast<uint32_t>(input.total() * input.elemSize());
    rknn_in.buf = input.data;
    rknn_in.pass_through = 0;
    int rc = rknn_inputs_set(impl_->context, 1, &rknn_in);
    if (rc == RKNN_SUCC) rc = rknn_run(impl_->context, nullptr);
    if (rc != RKNN_SUCC) { impl_->error = "RKNN execution failed: " + std::to_string(rc); return rc; }

    std::vector<rknn_output> rknn_out(impl_->io_count.n_output);
    for (uint32_t i = 0; i < impl_->io_count.n_output; ++i) {
        rknn_out[i] = {};
        rknn_out[i].index = i;
        rknn_out[i].want_float = 1;
    }
    rc = rknn_outputs_get(impl_->context, impl_->io_count.n_output, rknn_out.data(), nullptr);
    if (rc != RKNN_SUCC) { impl_->error = "RKNN output retrieval failed: " + std::to_string(rc); return rc; }
    tensors->reserve(rknn_out.size());
    for (std::size_t i = 0; i < rknn_out.size(); ++i) {
        TensorView tensor;
        tensor.nchw = impl_->output_attrs[i].fmt == RKNN_TENSOR_NCHW;
        if (impl_->output_attrs[i].fmt == RKNN_TENSOR_UNDEFINED && impl_->output_attrs[i].n_dims == 4) {
            const auto* dims = impl_->output_attrs[i].dims;
            tensor.nchw = dims[1] <= 256 && dims[2] == dims[3] && dims[1] != dims[2];
        }
        for (uint32_t d = 0; d < impl_->output_attrs[i].n_dims; ++d)
            tensor.dims.push_back(static_cast<int>(impl_->output_attrs[i].dims[d]));
        const std::size_t count = impl_->output_attrs[i].n_elems;
        const float* values = static_cast<const float*>(rknn_out[i].buf);
        tensor.values.assign(values, values + count);
        tensors->push_back(std::move(tensor));
    }
    rknn_outputs_release(impl_->context, impl_->io_count.n_output, rknn_out.data());
    if (transform) {
        *transform = LetterboxTransform{frame->width, frame->height, impl_->input_width,
                                        impl_->input_height, scale, pad_x, pad_y};
    }
    return 0;
#endif
}

void RknnModel::SetThresholds(float confidence, float nms) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->confidence = std::clamp(confidence, 0.0f, 1.0f);
    impl_->nms = std::clamp(nms, 0.0f, 1.0f);
}

int RknnModel::Deinit() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef HAVE_RKNN
    if (impl_->initialized) rknn_destroy(impl_->context);
    impl_->context = 0;
#endif
    impl_->initialized = false;
    return 0;
}

const std::string& RknnModel::lastError() const { return impl_->error; }
bool RknnModel::available() const { return impl_->initialized; }

}  // namespace rkdl
