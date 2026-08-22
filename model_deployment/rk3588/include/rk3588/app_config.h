#pragma once

#include <set>
#include <string>
#include <vector>

struct ModelConfig {
    std::string name;
    std::string path;
    float confidence = 0.45f;
    float nms = 0.45f;
    bool roi_from_previous = false;
};

struct InputConfig {
    int stream_id = 0;
    std::string source;
    std::string input_codec = "auto";
    int output_width = 1920;
    int output_height = 1080;
    int fps = 30;
    bool enable_ai = true;
    bool enable_raw = false;
    bool command_line_model = false;
    std::string mediamtx_endpoint;
    std::vector<ModelConfig> models;
};

struct AppConfig {
    std::vector<InputConfig> inputs;
    std::string dynamic_config_path = "/dev/shm/ai_config.json";
    std::string alarm_rules_path = "../config/alarm_rules.json";
    std::string backend_url = "http://127.0.0.1:8001";
    int device_id = 1;
    bool show_help = false;
};

bool parse_app_config(int argc, char** argv, AppConfig& config, std::string& error);
std::string usage_text(const char* program);
std::string resolve_rknn_model(const std::string& name_or_path);
