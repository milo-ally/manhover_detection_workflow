#include "rk3588/app_config.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include "json.hpp"

namespace {
bool is_option(const char* value) { return value && value[0] == '-'; }

bool split_endpoint(const std::string& endpoint, std::string& host, int& port) {
    const auto colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == endpoint.size()) return false;
    host = endpoint.substr(0, colon);
    char* end = nullptr;
    const long parsed = std::strtol(endpoint.c_str() + colon + 1, &end, 10);
    if (!end || *end || parsed < 1 || parsed > 65535) return false;
    port = static_cast<int>(parsed);
    return true;
}

template <typename T>
T value_or(const nlohmann::json& object, const char* key, T fallback) {
    return object.contains(key) ? object[key].get<T>() : fallback;
}
}

std::string resolve_rknn_model(const std::string& name_or_path) {
    if (name_or_path.empty() || name_or_path == "none") return {};
    std::filesystem::path path(name_or_path);
    if (path.extension() == ".rknn") return name_or_path;
    if (path.extension() == ".axmodel" || path.extension() == ".onnx") path.replace_extension(".rknn");
    if (path.has_parent_path()) return path.string();
    std::string name = path.stem().string();
    if (name == "pose" || name == "fall" || name == "behavior") name = "yolo11_pose_cut";
    else if (name == "helmet" || name == "yolo") name = "yolo11_helmet";
    else if (name == "crowd" || name == "human" || name == "human_detection") name = "yolo11_human_detection";
    else if (name == "face" || name == "face_detection") name = "face_detector_cut";
    else if (name == "face_recognition") name = "arcface_model2";
    return "../models/" + name + ".rknn";
}

std::string usage_text(const char* program) {
    std::ostringstream text;
    text << "Usage: " << (program ? program : "demo_helmet")
         << " [input ...] [-m INDEX:MODEL] [-c streams.json]"
            " [--mediamtx HOST:PORT] [--mediamtx-host HOST]"
            " [--mediamtx-port PORT] [--enable-raw [INDEX,...]]\n";
    return text.str();
}

bool parse_app_config(int argc, char** argv, AppConfig& config, std::string& error) {
    std::map<int, std::string> model_overrides;
    std::set<int> raw_indices;
    bool raw_all = false;
    bool endpoint_from_cli = false;
    std::string config_path, endpoint, host, port;
    std::vector<std::string> sources;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { config.show_help = true; return true; }
        if (arg == "-c" || arg == "--mediamtx" || arg == "--mediamtx-host" || arg == "--mediamtx-port" || arg == "-m") {
            if (++i >= argc) { error = "missing value after " + arg; return false; }
            const std::string value = argv[i];
            if (arg == "-c") config_path = value;
            else if (arg == "--mediamtx") { endpoint = value; endpoint_from_cli = true; }
            else if (arg == "--mediamtx-host") { host = value; endpoint_from_cli = true; }
            else if (arg == "--mediamtx-port") { port = value; endpoint_from_cli = true; }
            else {
                const auto colon = value.find(':');
                if (colon == std::string::npos) { error = "-m expects INDEX:MODEL"; return false; }
                model_overrides[std::stoi(value.substr(0, colon))] = value.substr(colon + 1);
            }
        } else if (arg == "--enable-raw") {
            if (i + 1 < argc && !is_option(argv[i + 1])) {
                const std::string value = argv[i + 1];
                if (value.find_first_not_of("0123456789,") == std::string::npos) {
                    ++i;
                    std::stringstream list(value);
                    std::string item;
                    while (std::getline(list, item, ',')) if (!item.empty()) raw_indices.insert(std::stoi(item));
                } else raw_all = true;
            } else raw_all = true;
        } else if (is_option(argv[i])) {
            error = "unknown option: " + arg;
            return false;
        } else sources.push_back(arg);
    }

    const char* env_host = std::getenv("MEDIAMTX_HOST");
    const char* env_port = std::getenv("MEDIAMTX_RTP_PORT");
    const bool endpoint_from_environment = env_host || env_port;
    if (endpoint.empty()) {
        if (host.empty()) host = env_host ? env_host : "127.0.0.1";
        if (port.empty()) port = env_port ? env_port : "8000";
        endpoint = host + ":" + port;
    }
    std::string endpoint_host;
    int base_port = 0;
    if (!split_endpoint(endpoint, endpoint_host, base_port)) { error = "invalid MediaMTX endpoint: " + endpoint; return false; }

    if (!config_path.empty()) {
        std::ifstream file(config_path);
        if (!file) { error = "cannot open streams config: " + config_path; return false; }
        try {
            nlohmann::json root; file >> root;
            if (!root.contains("streams") || !root["streams"].is_array()) { error = "streams config has no streams array"; return false; }
            if (!endpoint_from_cli && !endpoint_from_environment && root.contains("global_settings")) {
                const auto& global = root["global_settings"];
                endpoint_host = value_or<std::string>(global, "mediamtx_host", endpoint_host);
                base_port = std::stoi(value_or<std::string>(global, "mediamtx_port", std::to_string(base_port)));
            }
            const bool global_raw = root.contains("global_settings")
                ? value_or<bool>(root["global_settings"], "enable_raw_stream", false)
                : false;
            int index = 0;
            for (const auto& item : root["streams"]) {
                InputConfig input;
                input.stream_id = value_or<int>(item, "stream_id", index + 1);
                input.source = value_or<std::string>(item, "input_source", "");
                input.input_codec = value_or<std::string>(item, "input_codec", "auto");
                input.output_width = value_or<int>(item, "output_width", 1920);
                input.output_height = value_or<int>(item, "output_height", 1080);
                input.fps = value_or<int>(item, "fps", 30);
                input.enable_ai = value_or<bool>(item, "enable_ai", true);
                input.enable_raw = value_or<bool>(item, "enable_raw_stream", global_raw);
                input.mediamtx_endpoint = endpoint_host + ":" + std::to_string(base_port + index * 2);
                if (item.contains("models") && item["models"].is_array()) {
                    for (const auto& m : item["models"]) {
                        ModelConfig model;
                        model.name = value_or<std::string>(m, "name", "default");
                        model.path = resolve_rknn_model(value_or<std::string>(m, "path", model.name));
                        model.confidence = value_or<float>(m, "conf_threshold", 0.45f);
                        model.nms = value_or<float>(m, "nms_threshold", 0.45f);
                        model.roi_from_previous = value_or<bool>(m, "roi_from_previous", false);
                        input.models.push_back(std::move(model));
                    }
                } else if (item.contains("model_name")) {
                    ModelConfig model;
                    model.name = item["model_name"].get<std::string>();
                    model.path = resolve_rknn_model(model.name);
                    model.confidence = value_or<float>(item, "conf_thres", 0.45f);
                    model.nms = value_or<float>(item, "nms_thres", 0.45f);
                    input.models.push_back(std::move(model));
                }
                if (input.source.empty()) { error = "stream input_source is empty"; return false; }
                config.inputs.push_back(std::move(input));
                ++index;
            }
        } catch (const std::exception& e) { error = std::string("invalid streams config: ") + e.what(); return false; }
    } else {
        if (sources.empty()) sources.push_back("test.h264");
        for (std::size_t i = 0; i < sources.size(); ++i) {
            InputConfig input;
            input.stream_id = static_cast<int>(i) * 2 + 1;
            input.source = sources[i];
            input.enable_raw = raw_all || raw_indices.count(static_cast<int>(i));
            input.mediamtx_endpoint = endpoint_host + ":" + std::to_string(base_port + static_cast<int>(i) * 2);
            const auto model = model_overrides.find(static_cast<int>(i));
            if (model != model_overrides.end()) {
                input.command_line_model = true;
                input.models.push_back({model->second, resolve_rknn_model(model->second), 0.45f, 0.45f, false});
            }
            config.inputs.push_back(std::move(input));
        }
    }
    if (const char* value = std::getenv("BACKEND_API_URL")) config.backend_url = value;
    if (const char* value = std::getenv("DEVICE_ID")) config.device_id = std::atoi(value);
    return true;
}
