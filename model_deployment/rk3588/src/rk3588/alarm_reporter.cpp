#include "rk3588/alarm_reporter.h"

#include "http_client.h"
#include "json.hpp"
#include <fstream>
#include <cstring>

AlarmReporter::AlarmReporter(std::string backend_url, int device_id, const std::string& rules_path)
    : backend_url_(std::move(backend_url)), device_id_(device_id) {
    try {
        std::ifstream input(rules_path);
        nlohmann::json root; input >> root;
        if (root.contains("alarm_rules")) for (const auto& entry : root["alarm_rules"]) {
            Rule rule;
            rule.model_type = entry.value("model_type", "");
            rule.alarm_type = entry.value("alarm_type", rule.model_type);
            rule.severity = entry.value("severity", "high");
            rule.report_all = entry.value("report_all", false);
            if (entry.contains("labels")) for (const auto& label : entry["labels"])
                rule.labels.push_back(label.get<std::string>());
            if (!rule.model_type.empty()) rules_.push_back(std::move(rule));
        }
    } catch (...) {
        // Missing rules preserve the original built-in helmet/fall behavior.
    }
    thread_ = std::thread(&AlarmReporter::run, this);
}

AlarmReporter::~AlarmReporter() {
    { std::lock_guard<std::mutex> lock(mutex_); running_ = false; queue_.clear(); }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void AlarmReporter::submit(int camera_id, const std::string& model_name, const AI_RESULT_T& result) {
    if (result.nObjSize == 0) return;
    const Rule* selected = nullptr;
    for (const auto& rule : rules_) if (model_name.find(rule.model_type) != std::string::npos) {
        selected = &rule; break;
    }
    Rule fallback;
    if (!selected) {
        if (model_name.find("helmet") != std::string::npos) {
            fallback = {"helmet", "未戴安全帽", "high", {"no-helmet"}, false};
        } else if (model_name.find("fall") != std::string::npos || model_name.find("pose") != std::string::npos) {
            fallback = {"fall", "人員跌倒", "high", {}, true};
        } else return;
        selected = &fallback;
    }
    std::vector<const AI_OBJ_T*> objects;
    for (std::uint32_t i = 0; i < result.nObjSize; ++i) {
        if (selected->report_all) objects.push_back(&result.objects[i]);
        else for (const auto& label : selected->labels) if (std::strcmp(result.objects[i].label, label.c_str()) == 0) {
            objects.push_back(&result.objects[i]); break;
        }
    }
    if (objects.empty()) return;
    nlohmann::json body;
    body["device_id"] = device_id_;
    body["camera_id"] = camera_id;
    body["model_type"] = selected->model_type;
    body["alarm_type"] = selected->alarm_type;
    body["severity"] = selected->severity;
    body["detections"] = nlohmann::json::array();
    for (const AI_OBJ_T* pointer : objects) {
        const auto& object = *pointer;
        body["detections"].push_back({{"x", object.x}, {"y", object.y}, {"w", object.w},
                                       {"h", object.h}, {"score", object.score},
                                       {"label", object.label}, {"class_id", object.class_id},
                                       {"track_id", object.track_id}});
    }
    { 
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= 32) queue_.pop_front();
        queue_.push_back(body.dump());
    }
    cv_.notify_one();
}

void AlarmReporter::run() {
    while (true) {
        std::string payload;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return !running_ || !queue_.empty(); });
            if (!running_ && queue_.empty()) return;
            payload = std::move(queue_.front());
            queue_.pop_front();
        }
        HttpClient::post(backend_url_ + "/api/alarms", payload, 5);
    }
}
