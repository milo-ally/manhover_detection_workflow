#include <atomic>
#include <csignal>
#include <cstdio>
#include "rk3588/app_config.h"
#include "rk3588/application.h"

namespace {
Application* active_application = nullptr;
void stop_handler(int) {
    if (active_application) active_application->requestStop();
}
}

int main(int argc, char** argv) {
    AppConfig config;
    std::string error;
    if (!parse_app_config(argc, argv, config, error)) {
        std::fprintf(stderr, "Error: %s\n%s", error.c_str(), usage_text(argv[0]).c_str());
        return 2;
    }
    if (config.show_help) {
        std::fputs(usage_text(argv[0]).c_str(), stdout);
        return 0;
    }
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, stop_handler);
    std::signal(SIGTERM, stop_handler);
    Application application(std::move(config));
    active_application = &application;
    const int result = application.run();
    active_application = nullptr;
    return result;
}
