#include "rk3588/application.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include "rk3588/alarm_reporter.h"
#include "ffmpeg_pipeline.h"
#include "json.hpp"
#include "rtp_output.h"

namespace {

struct LoadedModel {
    ModelConfig config;
    IAIModel* model = nullptr;
    void* plugin = nullptr;
    DestroyAIModelFunc destroy = nullptr;
    std::chrono::steady_clock::time_point last_alarm{};
    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&& other) noexcept
        : config(std::move(other.config)), model(other.model), plugin(other.plugin), destroy(other.destroy),
          last_alarm(other.last_alarm) {
        other.model = nullptr; other.plugin = nullptr; other.destroy = nullptr;
    }
    LoadedModel& operator=(LoadedModel&& other) noexcept {
        if (this != &other) {
            reset(); config = std::move(other.config); model = other.model; plugin = other.plugin;
            destroy = other.destroy; last_alarm = other.last_alarm;
            other.model = nullptr; other.plugin = nullptr; other.destroy = nullptr;
        }
        return *this;
    }
    ~LoadedModel() { reset(); }
    void reset() {
        if (model) { model->Deinit(); if (destroy) destroy(model); }
        if (plugin) dlclose(plugin);
        model = nullptr; plugin = nullptr; destroy = nullptr;
    }
    bool available() const { return model != nullptr; }
};

std::string plugin_name(const std::string& hint) {
    if (hint.find("pose") != std::string::npos || hint.find("fall") != std::string::npos) return "libfall_plugin.so";
    if (hint.find("helmet") != std::string::npos) return "libhelmet_plugin.so";
    if (hint.find("fire") != std::string::npos || hint.find("smoke") != std::string::npos) return "libsmoke_fire_plugin.so";
    if (hint.find("plate") != std::string::npos) return "libplate_detection_plugin.so";
    // 井盖检测模型使用独立的 RKNN 插件策略
    if (hint.find("manhole") != std::string::npos || hint.find("manhole_cover") != std::string::npos) return "libmanhole_plugin.so";
    if (hint.find("crowd") != std::string::npos || hint.find("group") != std::string::npos) return "libcrowd_plugin.so";
    if (hint.find("face_rec") != std::string::npos || hint.find("arcface") != std::string::npos || hint.find("recognition") != std::string::npos) return "libface_recognition_plugin.so";
    if (hint.find("face") != std::string::npos) return "libface_detection_plugin.so";
    if (hint.find("human") != std::string::npos) return "libhuman_detection_plugin.so";
    if (hint.find("behavior") != std::string::npos) return "libbehavior_plugin.so";
    return "libyolo_plugin.so";
}

void* open_plugin(const std::string& name) {
    if (void* handle = dlopen(("./" + name).c_str(), RTLD_NOW | RTLD_LOCAL)) return handle;
    char executable[PATH_MAX]{};
    const ssize_t count = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (count > 0) {
        executable[count] = '\0';
        std::string path(executable);
        const auto slash = path.rfind('/');
        if (slash != std::string::npos)
            if (void* handle = dlopen((path.substr(0, slash + 1) + name).c_str(), RTLD_NOW | RTLD_LOCAL)) return handle;
    }
    return nullptr;
}

std::string runtime_path(const std::string& configured) {
    std::error_code error;
    if (std::filesystem::exists(configured, error)) return configured;
    char executable[PATH_MAX]{};
    const ssize_t count = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (count > 0) {
        executable[count] = '\0';
        std::filesystem::path candidate = std::filesystem::path(executable).parent_path() / configured;
        if (std::filesystem::exists(candidate, error)) return candidate.lexically_normal().string();
    }
    return configured;
}

std::string increment_port(const std::string& endpoint) {
    const auto colon = endpoint.rfind(':');
    if (colon == std::string::npos) return endpoint;
    return endpoint.substr(0, colon + 1) + std::to_string(std::stoi(endpoint.substr(colon + 1)) + 1);
}

void draw_result(cv::Mat& image, const AI_RESULT_T& result) {
    static const cv::Scalar colors[] = {{0, 220, 0}, {0, 80, 255}, {255, 180, 0}, {255, 0, 180}};
    for (std::uint32_t i = 0; i < result.nObjSize; ++i) {
        const auto& object = result.objects[i];
        const int x = std::clamp(static_cast<int>(object.x * image.cols), 0, image.cols - 1);
        const int y = std::clamp(static_cast<int>(object.y * image.rows), 0, image.rows - 1);
        const int width = std::clamp(static_cast<int>(object.w * image.cols), 1, image.cols - x);
        const int height = std::clamp(static_cast<int>(object.h * image.rows), 1, image.rows - y);
        const cv::Scalar color = colors[static_cast<unsigned>(object.class_id) % 4];
        cv::rectangle(image, cv::Rect(x, y, width, height), color, 2);
        char caption[96];
        std::snprintf(caption, sizeof(caption), "%s %.0f%%", object.label, object.score * 100.0f);
        cv::putText(image, caption, cv::Point(x, std::max(18, y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
        for (std::uint32_t k = 0; k < object.nKeypoints; ++k) {
            if (object.keypoints[k].conf < 0.25f) continue;
            cv::circle(image, cv::Point(static_cast<int>(object.keypoints[k].x * image.cols),
                                        static_cast<int>(object.keypoints[k].y * image.rows)), 3, color, -1);
        }
    }
}

std::vector<LoadedModel> load_models(const std::vector<ModelConfig>& configs) {
    std::vector<LoadedModel> loaded;
    for (const auto& config : configs) {
        LoadedModel item;
        item.config = config;
        const std::string model_path = runtime_path(config.path);
        const std::string library = plugin_name(config.name.empty() ? config.path : config.name);
        item.plugin = open_plugin(library);
        if (!item.plugin) {
            std::fprintf(stderr, "[RKNN] cannot load %s: %s\n", library.c_str(), dlerror());
        } else {
            auto create = reinterpret_cast<CreateAIModelFunc>(dlsym(item.plugin, "CreateAIModel"));
            item.destroy = reinterpret_cast<DestroyAIModelFunc>(dlsym(item.plugin, "DestroyAIModel"));
            if (create && item.destroy) item.model = create();
            if (!item.model || item.model->Init(model_path.c_str()) != 0) {
                std::fprintf(stderr, "[RKNN] model '%s' unavailable (%s)\n", config.name.c_str(), model_path.c_str());
                item.reset();
            } else {
                item.model->SetThresholds(config.confidence, config.nms);
                std::fprintf(stdout, "[RKNN] loaded %s with %s\n", model_path.c_str(), library.c_str());
            }
        }
        loaded.push_back(std::move(item));
    }
    return loaded;
}

void append_result(AI_RESULT_T& target, const AI_RESULT_T& source) {
    for (std::uint32_t i = 0; i < source.nObjSize && target.nObjSize < MAX_DETECT_OBJ_NUM; ++i)
        target.objects[target.nObjSize++] = source.objects[i];
}

bool read_dynamic_models(const std::string& path, int stream_id, bool model_locked,
                         std::vector<ModelConfig>& configs, std::filesystem::file_time_type& observed) {
    std::error_code ec;
    const auto modified = std::filesystem::last_write_time(path, ec);
    if (ec || modified == observed) return false;
    observed = modified;
    try {
        std::ifstream file(path);
        nlohmann::json root; file >> root;
        const nlohmann::json* selected = &root;
        if (root.contains("streams") && root["streams"].is_array()) {
            selected = nullptr;
            for (const auto& stream : root["streams"]) {
                const int id = stream.value("stream_id", stream.value("camera_id", -1));
                if (id == stream_id) { selected = &stream; break; }
            }
            if (!selected) return false;
        }
        const float global_conf = selected->value("conf_thres", 0.45f);
        const float global_nms = selected->value("nms_thres", 0.45f);
        if (model_locked) {
            for (auto& config : configs) { config.confidence = global_conf; config.nms = global_nms; }
            return true;
        }
        std::vector<ModelConfig> next;
        if (selected->contains("models") && (*selected)["models"].is_array()) {
            for (const auto& entry : (*selected)["models"]) {
                ModelConfig model;
                model.name = entry.value("name", "default");
                model.path = resolve_rknn_model(entry.value("path", model.name));
                model.confidence = entry.value("conf_threshold", global_conf);
                model.nms = entry.value("nms_threshold", global_nms);
                model.roi_from_previous = entry.value("roi_from_previous", false);
                next.push_back(std::move(model));
            }
        } else {
            const std::string name = selected->value("model_name", std::string{});
            if (!name.empty() && name != "none") next.push_back({name, resolve_rknn_model(name), global_conf, global_nms, false});
        }
        configs = std::move(next);
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[Config] ignored incomplete update: %s\n", e.what());
        return false;
    }
}

void run_input(InputConfig input, const AppConfig& app, std::atomic<bool>& stop, AlarmReporter& alarms) {
    std::vector<LoadedModel> models = load_models(input.models);
    std::filesystem::file_time_type dynamic_mtime{};
    const bool network_source = input.source.rfind("rtsp://", 0) == 0 || input.source.rfind("http://", 0) == 0;
    do {
        rkmedia::VideoSource source;
        if (!source.open(input.source, input.input_codec)) {
            std::fprintf(stderr, "[Stream %d] %s\n", input.stream_id, source.lastError().c_str());
            if (!network_source || stop.load()) return;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        std::fprintf(stdout, "[Stream %d] decoder=%s input=%s\n", input.stream_id,
                     source.usingRockchipDecoder() ? "rkmpp" : "software", input.source.c_str());
        rkmedia::RtpOutput output, raw_output;
        bool output_open = false, raw_open = false;
        std::uint64_t frame_number = 0;
        AI_RESULT_T latest_result{};
        source.run([&](rkmedia::DecodedFrame&& decoded) {
            if (stop.load()) return false;
            if (!output_open) {
                output_open = output.open(input.mediamtx_endpoint, input.output_width, input.output_height, input.fps);
                if (!output_open) std::fprintf(stderr, "[Stream %d] output: %s\n", input.stream_id, output.lastError().c_str());
                else std::fprintf(stdout, "[Stream %d] encoder=%s rtp=%s\n", input.stream_id,
                                  output.usingRockchipEncoder() ? "rkmpp" : "software", input.mediamtx_endpoint.c_str());
            }
            if (input.enable_raw && !raw_open) {
                const std::string raw_endpoint = increment_port(input.mediamtx_endpoint);
                raw_open = raw_output.open(raw_endpoint, input.output_width, input.output_height, input.fps);
                if (raw_open) std::fprintf(stdout, "[Stream %d] raw rtp=%s\n", input.stream_id, raw_endpoint.c_str());
            }
            if (raw_open) raw_output.write(decoded.bgr, static_cast<std::int64_t>(frame_number));

            if (++frame_number % 15 == 0) {
                auto current = input.models;
                if (read_dynamic_models(app.dynamic_config_path, input.stream_id, input.command_line_model,
                                        current, dynamic_mtime)) {
                    input.models = std::move(current);
                    models = load_models(input.models);
                    latest_result = {};
                    std::fprintf(stdout, "[Config] stream %d reloaded %zu model(s)\n", input.stream_id, models.size());
                }
            }
            if (input.enable_ai && !models.empty() && frame_number % std::max(1, input.fps / 15) == 0) {
                VideoFrame frame;
                frame.width = decoded.bgr.cols;
                frame.height = decoded.bgr.rows;
                frame.stride = static_cast<int>(decoded.bgr.step);
                frame.format = PixelFormat::BGR888;
                frame.data = decoded.bgr.data;
                frame.size = decoded.bgr.total() * decoded.bgr.elemSize();
                frame.pts = decoded.pts;
                AI_RESULT_T merged{};
                AI_RESULT_T previous{};
                for (auto& item : models) {
                    if (!item.available()) continue;
                    AI_RESULT_T result{};
                    int inference_rc = 0;
                    if (item.config.roi_from_previous && previous.nObjSize > 0) {
                        for (std::uint32_t roi_index = 0; roi_index < previous.nObjSize &&
                             result.nObjSize < MAX_DETECT_OBJ_NUM; ++roi_index) {
                            const auto& roi = previous.objects[roi_index];
                            cv::Rect rectangle(static_cast<int>(roi.x * decoded.bgr.cols),
                                               static_cast<int>(roi.y * decoded.bgr.rows),
                                               static_cast<int>(roi.w * decoded.bgr.cols),
                                               static_cast<int>(roi.h * decoded.bgr.rows));
                            rectangle &= cv::Rect(0, 0, decoded.bgr.cols, decoded.bgr.rows);
                            if (rectangle.width < 2 || rectangle.height < 2) continue;
                            cv::Mat crop = decoded.bgr(rectangle);
                            VideoFrame crop_frame;
                            crop_frame.width = crop.cols; crop_frame.height = crop.rows;
                            crop_frame.stride = static_cast<int>(crop.step);
                            crop_frame.format = PixelFormat::BGR888; crop_frame.data = crop.data;
                            crop_frame.size = crop.total() * crop.elemSize(); crop_frame.pts = decoded.pts;
                            AI_RESULT_T crop_result{};
                            inference_rc = item.model->Inference(&crop_frame, &crop_result);
                            if (inference_rc != 0) break;
                            for (std::uint32_t k = 0; k < crop_result.nObjSize && result.nObjSize < MAX_DETECT_OBJ_NUM; ++k) {
                                AI_OBJ_T object = crop_result.objects[k];
                                object.x = roi.x + object.x * roi.w;
                                object.y = roi.y + object.y * roi.h;
                                object.w *= roi.w; object.h *= roi.h;
                                result.objects[result.nObjSize++] = object;
                            }
                        }
                    } else {
                        inference_rc = item.model->Inference(&frame, &result);
                    }
                    if (inference_rc == 0) {
                        append_result(merged, result);
                        previous = result;
                        const auto now = std::chrono::steady_clock::now();
                        if (result.nObjSize && now - item.last_alarm > std::chrono::seconds(2)) {
                            alarms.submit(input.stream_id, item.config.name, result);
                            item.last_alarm = now;
                        }
                    }
                }
                latest_result = merged;
            }
            if (input.enable_ai) draw_result(decoded.bgr, latest_result);
            if (output_open && !output.write(decoded.bgr, static_cast<std::int64_t>(frame_number))) {
                std::fprintf(stderr, "[Stream %d] output failed: %s\n", input.stream_id, output.lastError().c_str());
                output.close(); output_open = false;
            }
            return true;
        }, stop);
        if (!network_source) break;
        if (!stop.load()) {
            std::fprintf(stderr, "[Stream %d] input disconnected, reconnecting\n", input.stream_id);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } while (!stop.load());
}

}  // namespace

Application::Application(AppConfig config) : config_(std::move(config)) {}
void Application::requestStop() { stop_requested_.store(true); }

int Application::run() {
    AlarmReporter reporter(config_.backend_url, config_.device_id, runtime_path(config_.alarm_rules_path));
    std::vector<std::thread> workers;
    for (const auto& input : config_.inputs)
        workers.emplace_back(run_input, input, std::cref(config_), std::ref(stop_requested_), std::ref(reporter));
    for (auto& worker : workers) worker.join();
    return 0;
}
