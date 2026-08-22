#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "ai_interface.h"

class AlarmReporter {
public:
    AlarmReporter(std::string backend_url, int device_id, const std::string& rules_path);
    ~AlarmReporter();
    void submit(int camera_id, const std::string& model_name, const AI_RESULT_T& result);

private:
    struct Rule {
        std::string model_type;
        std::string alarm_type;
        std::string severity;
        std::vector<std::string> labels;
        bool report_all = false;
    };
    void run();
    std::string backend_url_;
    int device_id_;
    bool running_ = true;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::string> queue_;
    std::thread thread_;
    std::vector<Rule> rules_;
};
