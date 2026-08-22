#include "rknn_model.h"
#include "face_aligner.h"
#include "bytetrack.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits.h>
#include <unistd.h>

#ifndef RK_PLUGIN_NAME
#define RK_PLUGIN_NAME "default"
#endif

namespace {

class PolicyModel final : public IAIModel {
public:
    PolicyModel() : profile_(std::string(RK_PLUGIN_NAME) == "yolo" ? "helmet" : RK_PLUGIN_NAME),
                    runner_(profile_), tracker_(bytetracker_create(30, 30)) {}
    ~PolicyModel() override { Deinit(); }
    int Init(const char* path) override {
        const int rc = runner_.Init(path);
        if (rc == 0 && profile_ == "face_recognition") {
            load_gallery();
            const char* configured = std::getenv("FACE_LANDMARK_PATH");
            std::string landmark = configured && *configured ? configured : "../models/shape_predictor_68_face_landmarks.dat";
            if (!aligner_.load(landmark) && (!configured || !*configured)) {
                char executable[PATH_MAX]{};
                const ssize_t count = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
                if (count > 0) {
                    executable[count] = '\0'; std::string location(executable);
                    const auto slash = location.rfind('/');
                    if (slash != std::string::npos)
                        aligner_.load(location.substr(0, slash + 1) + "../models/shape_predictor_68_face_landmarks.dat");
                }
            }
        }
        return rc;
    }
    void GetInputSize(int* width, int* height) override { runner_.GetInputSize(width, height); }
    void SetThresholds(float confidence, float nms) override { runner_.SetThresholds(confidence, nms); }
    int Deinit() override {
        tracks_.clear(); gallery_.clear();
        if (tracker_) bytetracker_release(&tracker_);
        return runner_.Deinit();
    }

    int Inference(const VideoFrame* frame, AI_RESULT_T* result) override {
        if (profile_ == "face_recognition") return recognize(frame, result);
        const int rc = runner_.Inference(frame, result);
        if (rc != 0) return rc;
        assign_tracks(*result);
        if (profile_ == "crowd") aggregate_crowds(*result);
        else if (profile_ == "fall") classify_falls(*result);
        return 0;
    }

private:
    struct FallTrack { int state = 0; int votes = 0; int lost = 0; float cx = 0, cy = 0; };
    struct Identity { std::string name; std::vector<float> embedding; };

    void assign_tracks(AI_RESULT_T& result) {
        if (!tracker_ || result.nObjSize == 0) return;
        bytetrack_object_t input{};
        input.n_objects = static_cast<int>(std::min<std::uint32_t>(result.nObjSize, TRACK_OBJETCS_MAX_SIZE));
        for (int i = 0; i < input.n_objects; ++i) {
            input.objects[i].label = result.objects[i].class_id;
            input.objects[i].prob = result.objects[i].score;
            input.objects[i].rect = {result.objects[i].x, result.objects[i].y,
                                     result.objects[i].w, result.objects[i].h};
        }
        bytetracker_track(tracker_, &input);
        for (std::uint32_t i = 0; i < result.nObjSize; ++i) {
            float best = 1e9f; long track = 0;
            const float cx = result.objects[i].x + result.objects[i].w * 0.5f;
            const float cy = result.objects[i].y + result.objects[i].h * 0.5f;
            for (int j = 0; j < input.n_track_objects; ++j) {
                if (input.track_objects[j].label != result.objects[i].class_id) continue;
                const float tx = input.track_objects[j].rect.x + input.track_objects[j].rect.width * 0.5f;
                const float ty = input.track_objects[j].rect.y + input.track_objects[j].rect.height * 0.5f;
                const float distance = (cx - tx) * (cx - tx) + (cy - ty) * (cy - ty);
                if (distance < best) { best = distance; track = input.track_objects[j].track_id; }
            }
            result.objects[i].track_id = static_cast<std::uint64_t>(std::max<long>(0, track));
        }
    }

    void load_gallery() {
        const char* configured = std::getenv("FACE_DB_PATH");
        std::string path = configured && *configured ? configured : "../models/known_faces_arcface.txt";
        std::ifstream input(path);
        if (!input && (!configured || !*configured)) {
            char executable[PATH_MAX]{};
            const ssize_t count = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
            if (count > 0) {
                executable[count] = '\0';
                std::string location(executable);
                const auto slash = location.rfind('/');
                if (slash != std::string::npos) {
                    input.clear();
                    input.open(location.substr(0, slash + 1) + "../models/known_faces_arcface.txt");
                }
            }
        }
        std::string line;
        while (std::getline(input, line)) {
            std::istringstream row(line); Identity identity;
            if (!(row >> identity.name)) continue;
            float value;
            while (row >> value) identity.embedding.push_back(value);
            if (!identity.embedding.empty()) gallery_.push_back(std::move(identity));
        }
        if (const char* threshold = std::getenv("FACE_REC_THRESHOLD")) recognition_threshold_ = std::atof(threshold);
    }

    int recognize(const VideoFrame* frame, AI_RESULT_T* result) {
        *result = {};
        if (gallery_.empty()) return 0;
        cv::Mat aligned;
        VideoFrame aligned_frame;
        const VideoFrame* inference_frame = frame;
        if (aligner_.align112Rgb(*frame, aligned)) {
            aligned_frame.width = aligned.cols; aligned_frame.height = aligned.rows;
            aligned_frame.stride = static_cast<int>(aligned.step);
            aligned_frame.format = PixelFormat::RGB888; aligned_frame.data = aligned.data;
            aligned_frame.size = aligned.total() * aligned.elemSize(); aligned_frame.pts = frame->pts;
            inference_frame = &aligned_frame;
        }
        std::vector<rkdl::TensorView> outputs;
        if (runner_.RunRaw(inference_frame, &outputs) != 0 || outputs.empty() || outputs[0].values.empty()) return -1;
        std::vector<float> embedding = outputs[0].values;
        double norm = 0.0;
        for (float value : embedding) norm += static_cast<double>(value) * value;
        if (norm <= 1e-12) return -1;
        const float inverse = 1.0f / static_cast<float>(std::sqrt(norm));
        for (float& value : embedding) value *= inverse;
        float best_score = -1.0f; std::string best_name = "unknown_face";
        for (const auto& identity : gallery_) {
            if (identity.embedding.size() != embedding.size()) continue;
            float score = 0.0f;
            for (std::size_t i = 0; i < embedding.size(); ++i) score += embedding[i] * identity.embedding[i];
            if (score > best_score) { best_score = score; best_name = identity.name; }
        }
        if (best_score < recognition_threshold_) best_name = "unknown_face";
        result->nObjSize = 1;
        auto& object = result->objects[0];
        object.x = 0; object.y = 0; object.w = 1; object.h = 1;
        object.score = best_score; object.class_id = best_name == "unknown_face" ? -1 : 0;
        std::snprintf(object.label, sizeof(object.label), "%s", best_name.c_str());
        return 0;
    }

    void classify_falls(AI_RESULT_T& result) {
        for (auto& entry : tracks_) entry.second.lost++;
        for (std::uint32_t i = 0; i < result.nObjSize; ++i) {
            auto& object = result.objects[i];
            const float cx = object.x + object.w * 0.5f, cy = object.y + object.h * 0.5f;
            int best = -1; float distance = 1e9f;
            for (auto& entry : tracks_) {
                const float dx = cx - entry.second.cx, dy = cy - entry.second.cy;
                const float candidate = dx * dx + dy * dy;
                if (candidate < distance && candidate < 0.04f) { distance = candidate; best = entry.first; }
            }
            if (best < 0) best = next_track_++;
            auto& track = tracks_[best];
            track.cx = cx; track.cy = cy; track.lost = 0;
            bool horizontal = object.w > object.h * 1.05f;
            if (object.nKeypoints >= 13) {
                const auto& left_shoulder = object.keypoints[5];
                const auto& right_shoulder = object.keypoints[6];
                const auto& left_hip = object.keypoints[11];
                const auto& right_hip = object.keypoints[12];
                if (left_shoulder.conf > 0.2f && right_shoulder.conf > 0.2f &&
                    left_hip.conf > 0.2f && right_hip.conf > 0.2f) {
                    const float sx = (left_shoulder.x + right_shoulder.x) * 0.5f;
                    const float sy = (left_shoulder.y + right_shoulder.y) * 0.5f;
                    const float hx = (left_hip.x + right_hip.x) * 0.5f;
                    const float hy = (left_hip.y + right_hip.y) * 0.5f;
                    horizontal = std::abs(hx - sx) > std::abs(hy - sy);
                }
            }
            track.votes = std::clamp(track.votes + (horizontal ? 1 : -1), 0, 8);
            if (track.votes >= 4) track.state = 2;
            else if (track.votes >= 2) track.state = 1;
            else track.state = 0;
            object.track_id = static_cast<std::uint64_t>(best);
            object.class_id = track.state;
            std::snprintf(object.label, sizeof(object.label), "%s",
                          track.state == 2 ? "FALLEN!" : track.state == 1 ? "falling?" : "standing");
        }
        for (auto it = tracks_.begin(); it != tracks_.end();) {
            if (it->second.lost > 30) it = tracks_.erase(it); else ++it;
        }
    }

    void aggregate_crowds(AI_RESULT_T& result) {
        constexpr float radius = 0.14f;
        constexpr int minimum_people = 3;
        const std::uint32_t count = result.nObjSize;
        std::vector<int> group(count, -1);
        int group_count = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            if (group[i] >= 0) continue;
            std::vector<std::uint32_t> queue{i};
            group[i] = group_count;
            for (std::size_t q = 0; q < queue.size(); ++q) {
                const auto a = queue[q];
                const float ax = result.objects[a].x + result.objects[a].w * 0.5f;
                const float ay = result.objects[a].y + result.objects[a].h;
                for (std::uint32_t j = 0; j < count; ++j) {
                    if (group[j] >= 0) continue;
                    const float bx = result.objects[j].x + result.objects[j].w * 0.5f;
                    const float by = result.objects[j].y + result.objects[j].h;
                    if (std::hypot(ax - bx, ay - by) <= radius) { group[j] = group_count; queue.push_back(j); }
                }
            }
            ++group_count;
        }
        AI_RESULT_T crowds{};
        for (int id = 0; id < group_count && crowds.nObjSize < MAX_DETECT_OBJ_NUM; ++id) {
            int members = 0; float x1 = 1, y1 = 1, x2 = 0, y2 = 0, score = 0;
            for (std::uint32_t i = 0; i < count; ++i) if (group[i] == id) {
                ++members; const auto& person = result.objects[i];
                x1 = std::min(x1, person.x); y1 = std::min(y1, person.y);
                x2 = std::max(x2, person.x + person.w); y2 = std::max(y2, person.y + person.h);
                score = std::max(score, person.score);
            }
            if (members < minimum_people) continue;
            auto& out = crowds.objects[crowds.nObjSize++];
            out.x = x1; out.y = y1; out.w = x2 - x1; out.h = y2 - y1; out.score = score;
            out.class_id = 0; std::snprintf(out.label, sizeof(out.label), "G (%d)", members);
        }
        result = crowds;
    }

    std::string profile_;
    rkdl::RknnModel runner_;
    std::map<int, FallTrack> tracks_;
    int next_track_ = 1;
    std::vector<Identity> gallery_;
    float recognition_threshold_ = 0.36f;
    bytetracker_t tracker_ = nullptr;
    rkdl::FaceAligner aligner_;
};

}  // namespace

extern "C" IAIModel* CreateAIModel() { return new PolicyModel(); }
extern "C" void DestroyAIModel(IAIModel* model) { delete model; }
