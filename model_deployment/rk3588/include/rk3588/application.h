#pragma once

#include <atomic>
#include "rk3588/app_config.h"

class Application {
public:
    explicit Application(AppConfig config);
    int run();
    void requestStop();

private:
    AppConfig config_;
    std::atomic<bool> stop_requested_{false};
};
