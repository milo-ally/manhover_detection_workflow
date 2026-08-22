#include "yolo_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace rkdl {
namespace {

struct Candidate {
    float x1, y1, x2, y2, score;
    int class_id;
    std::vector<AI_KEYPOINT_T> keypoints;
};

float overlap(const Candidate& a, const Candidate& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return intersection / std::max(1e-6f, area_a + area_b - intersection);
}

float sigmoid(float value) {
    return value < -20.0f ? 0.0f : value > 20.0f ? 1.0f : 1.0f / (1.0f + std::exp(-value));
}

float probability(float value) {
    return (value < 0.0f || value > 1.0f) ? sigmoid(value) : value;
}

const char* label_for(const std::string& profile, int class_id) {
    if (profile.find("helmet") != std::string::npos) {
        static const char* names[] = {"hat", "helmet", "no-helmet"};
        return class_id >= 0 && class_id < 3 ? names[class_id] : "object";
    }
    if (profile.find("construction") != std::string::npos) {
        static const char* names[] = {"person", "fire", "hat", "helmet", "vest", "safetyharness", "machiney", "smoke"};
        return class_id >= 0 && class_id < 8 ? names[class_id] : "object";
    }
    if (profile.find("behavior") != std::string::npos) {
        static const char* names[] = {"calling", "smoking"};
        return class_id >= 0 && class_id < 2 ? names[class_id] : "object";
    }
    if (profile.find("smoke") != std::string::npos) {
        static const char* names[] = {"smoke", "fire"};
        return class_id >= 0 && class_id < 2 ? names[class_id] : "object";
    }
    if (profile.find("manhole") != std::string::npos) {
        static const char* names[] = {"good", "broke", "lose", "uncovered", "circle"};
        return class_id >= 0 && class_id < 5 ? names[class_id] : "object";
    }
    if (profile.find("face") != std::string::npos) return "face";
    if (profile.find("plate") != std::string::npos) return "plate";
    if (profile.find("human") != std::string::npos || profile.find("crowd") != std::string::npos) return "person";
    if (profile.find("fall") != std::string::npos) return class_id == 0 ? "person" : "fall";
    return "object";
}

void undo_letterbox(Candidate& c, const LetterboxTransform& t) {
    const float scale = std::max(t.scale, 1e-6f);
    c.x1 = std::clamp((c.x1 - t.pad_x) / scale, 0.0f, static_cast<float>(t.source_width));
    c.y1 = std::clamp((c.y1 - t.pad_y) / scale, 0.0f, static_cast<float>(t.source_height));
    c.x2 = std::clamp((c.x2 - t.pad_x) / scale, 0.0f, static_cast<float>(t.source_width));
    c.y2 = std::clamp((c.y2 - t.pad_y) / scale, 0.0f, static_cast<float>(t.source_height));
    for (auto& p : c.keypoints) {
        p.x = std::clamp((p.x - t.pad_x) / scale / t.source_width, 0.0f, 1.0f);
        p.y = std::clamp((p.y - t.pad_y) / scale / t.source_height, 0.0f, 1.0f);
    }
}

struct FeatureMap {
    const TensorView* tensor = nullptr;
    int channels = 0;
    int height = 0;
    int width = 0;

    float at(int y, int x, int channel) const {
        if (tensor->nchw) {
            return tensor->values[(static_cast<std::size_t>(channel) * height + y) * width + x];
        }
        return tensor->values[(static_cast<std::size_t>(y) * width + x) * channels + channel];
    }
};

bool as_feature_map(const TensorView& tensor, FeatureMap& map) {
    if (tensor.dims.size() != 4 || tensor.dims[0] != 1) return false;
    map.tensor = &tensor;
    if (tensor.nchw) {
        map.channels = tensor.dims[1]; map.height = tensor.dims[2]; map.width = tensor.dims[3];
    } else {
        map.height = tensor.dims[1]; map.width = tensor.dims[2]; map.channels = tensor.dims[3];
    }
    return map.channels > 0 && map.height > 0 && map.width > 0 &&
           static_cast<std::size_t>(map.channels) * map.height * map.width <= tensor.values.size();
}

float dfl_distance(const FeatureMap& map, int y, int x, int side) {
    constexpr int bins = 16;
    float maximum = -1e30f;
    for (int bin = 0; bin < bins; ++bin) maximum = std::max(maximum, map.at(y, x, side * bins + bin));
    float total = 0.0f, weighted = 0.0f;
    for (int bin = 0; bin < bins; ++bin) {
        const float value = std::exp(map.at(y, x, side * bins + bin) - maximum);
        total += value; weighted += value * bin;
    }
    return weighted / std::max(total, 1e-12f);
}

void decode_feature_maps(const std::vector<TensorView>& outputs,
                         const LetterboxTransform& transform,
                         float confidence_threshold,
                         const std::string& profile,
                         std::vector<Candidate>& candidates) {
    std::vector<FeatureMap> maps;
    for (const auto& output : outputs) {
        FeatureMap map;
        if (as_feature_map(output, map)) maps.push_back(map);
    }
    const bool pose_profile = profile.find("fall") != std::string::npos || profile.find("pose") != std::string::npos ||
                              profile.find("behavior") != std::string::npos;
    const TensorView* global_keypoints = nullptr;
    bool keypoints_feature_first = true;
    int keypoint_cells = 0;
    if (pose_profile) for (const auto& output : outputs) {
        if (output.dims.size() != 3) continue;
        if (output.dims[1] == 51) {
            global_keypoints = &output; keypoints_feature_first = true; keypoint_cells = output.dims[2]; break;
        }
        if (output.dims[2] == 51) {
            global_keypoints = &output; keypoints_feature_first = false; keypoint_cells = output.dims[1]; break;
        }
    }
    int pose_cell_offset = 0;
    for (std::size_t map_index = 0; map_index < maps.size(); ++map_index) {
        const auto& map = maps[map_index];
        if (map.channels == 64) {
            // Official RKNN Model Zoo YOLO8/11 exports each branch as
            // [DFL box(64), class logits(N), optional score-sum(1)].
            const FeatureMap* classes = nullptr;
            if (map_index + 1 < maps.size() && maps[map_index + 1].height == map.height &&
                maps[map_index + 1].width == map.width && maps[map_index + 1].channels != 64 &&
                maps[map_index + 1].channels != 51) {
                classes = &maps[map_index + 1];
            }
            if (!classes) continue;
            const int stride_x = std::max(1, transform.network_width / map.width);
            const int stride_y = std::max(1, transform.network_height / map.height);
            for (int y = 0; y < map.height; ++y) for (int x = 0; x < map.width; ++x) {
                float score = 0.0f; int class_id = 0;
                for (int channel = 0; channel < classes->channels; ++channel) {
                    const float value = probability(classes->at(y, x, channel));
                    if (value > score) { score = value; class_id = channel; }
                }
                if (score < confidence_threshold) continue;
                const float center_x = (x + 0.5f) * stride_x;
                const float center_y = (y + 0.5f) * stride_y;
                Candidate candidate{
                    center_x - dfl_distance(map, y, x, 0) * stride_x,
                    center_y - dfl_distance(map, y, x, 1) * stride_y,
                    center_x + dfl_distance(map, y, x, 2) * stride_x,
                    center_y + dfl_distance(map, y, x, 3) * stride_y,
                    score, class_id, {}};
                undo_letterbox(candidate, transform);
                if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1)
                    candidates.push_back(std::move(candidate));
            }
            continue;
        }
        if (map.channels < 56 || map.channels == 51) continue;
        const bool direct_pose = map.channels == 56;
        const bool combined_pose = map.channels >= 116 && pose_profile;
        const int branch_keypoint_offset = pose_cell_offset;
        if (pose_profile && map.channels == 65) pose_cell_offset += map.height * map.width;
        const bool dfl = !direct_pose;
        const FeatureMap* keypoints = nullptr;
        if (pose_profile && !combined_pose) {
            for (const auto& candidate : maps) {
                if (candidate.height == map.height && candidate.width == map.width && candidate.channels == 51) {
                    keypoints = &candidate; break;
                }
            }
        }
        const int stride_x = std::max(1, transform.network_width / map.width);
        const int stride_y = std::max(1, transform.network_height / map.height);
        for (int y = 0; y < map.height; ++y) {
            for (int x = 0; x < map.width; ++x) {
                float score = 0.0f;
                int class_id = 0;
                int keypoint_start = -1;
                if (direct_pose) {
                    score = probability(map.at(y, x, 55));
                    keypoint_start = 4;
                } else if (combined_pose) {
                    score = probability(map.at(y, x, 115));
                    keypoint_start = 64;
                } else if (pose_profile && map.channels == 65) {
                    score = probability(map.at(y, x, 64));
                } else {
                    for (int channel = 64; channel < map.channels; ++channel) {
                        const float value = probability(map.at(y, x, channel));
                        if (value > score) { score = value; class_id = channel - 64; }
                    }
                }
                if (score < confidence_threshold) continue;
                float left, top, right, bottom;
                if (dfl) {
                    left = dfl_distance(map, y, x, 0) * stride_x;
                    top = dfl_distance(map, y, x, 1) * stride_y;
                    right = dfl_distance(map, y, x, 2) * stride_x;
                    bottom = dfl_distance(map, y, x, 3) * stride_y;
                } else {
                    left = map.at(y, x, 0) * stride_x;
                    top = map.at(y, x, 1) * stride_y;
                    right = map.at(y, x, 2) * stride_x;
                    bottom = map.at(y, x, 3) * stride_y;
                }
                const float center_x = (x + 0.5f) * stride_x;
                const float center_y = (y + 0.5f) * stride_y;
                Candidate candidate{center_x - left, center_y - top, center_x + right, center_y + bottom,
                                    score, class_id, {}};
                const FeatureMap* kp_map = keypoints ? keypoints : &map;
                const int kp_offset = keypoints ? 0 : keypoint_start;
                if (kp_offset >= 0) {
                    candidate.keypoints.reserve(MAX_KEYPOINTS);
                    for (int k = 0; k < MAX_KEYPOINTS; ++k) {
                        candidate.keypoints.push_back({
                            (kp_map->at(y, x, kp_offset + k * 3) * 2.0f + x) * stride_x,
                            (kp_map->at(y, x, kp_offset + k * 3 + 1) * 2.0f + y) * stride_y,
                            probability(kp_map->at(y, x, kp_offset + k * 3 + 2))});
                    }
                } else if (global_keypoints && map.channels == 65) {
                    const int cell = branch_keypoint_offset + y * map.width + x;
                    if (cell < keypoint_cells) {
                        auto keypoint_at = [&](int feature) {
                            return keypoints_feature_first
                                ? global_keypoints->values[static_cast<std::size_t>(feature) * keypoint_cells + cell]
                                : global_keypoints->values[static_cast<std::size_t>(cell) * 51 + feature];
                        };
                        candidate.keypoints.reserve(MAX_KEYPOINTS);
                        for (int k = 0; k < MAX_KEYPOINTS; ++k) {
                            float kx = keypoint_at(k * 3), ky = keypoint_at(k * 3 + 1);
                            if (std::abs(kx) <= 2.5f && std::abs(ky) <= 2.5f) {
                                kx = (kx * 2.0f + x) * stride_x;
                                ky = (ky * 2.0f + y) * stride_y;
                            }
                            candidate.keypoints.push_back({kx, ky, probability(keypoint_at(k * 3 + 2))});
                        }
                    }
                }
                undo_letterbox(candidate, transform);
                if (candidate.x2 > candidate.x1 && candidate.y2 > candidate.y1)
                    candidates.push_back(std::move(candidate));
            }
        }
    }
}

}  // namespace

int decode_yolo(const std::vector<TensorView>& outputs,
                const LetterboxTransform& transform,
                float confidence_threshold,
                float nms_threshold,
                const std::string& profile,
                AI_RESULT_T* result) {
    if (!result) return -1;
    *result = {};
    std::vector<Candidate> candidates;

    // RKNN Toolkit2 commonly exports YOLO11 heads as three NHWC/NCHW feature
    // maps (DFL box bins + class logits), and pose models may add three 51
    // channel keypoint maps.  Decode these before handling already-flattened
    // ONNX-style outputs.
    decode_feature_maps(outputs, transform, confidence_threshold, profile, candidates);

    for (const auto& tensor : outputs) {
        if (tensor.dims.empty() || tensor.values.empty()) continue;
        if (tensor.dims.size() > 3) continue;
        int rows = 0, features = 0;
        bool feature_first = false;
        if (tensor.dims.size() >= 2) {
            const int a = tensor.dims[tensor.dims.size() - 2];
            const int b = tensor.dims[tensor.dims.size() - 1];
            if (a >= 6 && a <= 256 && b > a) {
                features = a; rows = b; feature_first = true;
            } else if (b >= 6 && b <= 256) {
                rows = a; features = b;
            }
        }
        if (rows <= 0 || features < 6 || static_cast<std::size_t>(rows) * features > tensor.values.size()) continue;
        if (features == 51) continue;
        auto at = [&](int row, int col) -> float {
            return feature_first ? tensor.values[static_cast<std::size_t>(col) * rows + row]
                                 : tensor.values[static_cast<std::size_t>(row) * features + col];
        };

        const bool pose = features == 56 || profile.find("behavior") != std::string::npos || profile.find("fall") != std::string::npos;
        for (int row = 0; row < rows; ++row) {
            float score = 0.0f;
            int class_id = 0;
            int keypoint_offset = features;
            if (features == 6) {
                score = probability(at(row, 4));
                class_id = static_cast<int>(at(row, 5));
            } else if (pose && features >= 56) {
                score = probability(at(row, 4));
                keypoint_offset = 5;
            } else {
                // YOLOv8/11 has no explicit objectness; YOLOv5 does.  Select
                // the layout by its conventional feature count.
                const bool has_objectness = features == 85 || features == 25 || features == 8;
                const int class_begin = has_objectness ? 5 : 4;
                const float objectness = has_objectness ? probability(at(row, 4)) : 1.0f;
                for (int col = class_begin; col < features; ++col) {
                    const float value = probability(at(row, col)) * objectness;
                    if (value > score) { score = value; class_id = col - class_begin; }
                }
            }
            if (score < confidence_threshold) continue;

            const float cx = at(row, 0), cy = at(row, 1);
            const float width = at(row, 2), height = at(row, 3);
            Candidate c{cx - width * 0.5f, cy - height * 0.5f,
                        cx + width * 0.5f, cy + height * 0.5f, score, class_id, {}};
            if (keypoint_offset < features) {
                const int count = std::min(MAX_KEYPOINTS, (features - keypoint_offset) / 3);
                c.keypoints.reserve(count);
                for (int k = 0; k < count; ++k) {
                    c.keypoints.push_back({at(row, keypoint_offset + k * 3),
                                           at(row, keypoint_offset + k * 3 + 1),
                                           probability(at(row, keypoint_offset + k * 3 + 2))});
                }
            }
            undo_letterbox(c, transform);
            if (c.x2 > c.x1 && c.y2 > c.y1) candidates.push_back(std::move(c));
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.score > b.score;
    });
    std::vector<Candidate> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& selected : kept) {
            if (candidate.class_id == selected.class_id && overlap(candidate, selected) > nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
        if (kept.size() == MAX_DETECT_OBJ_NUM) break;
    }

    result->nObjSize = static_cast<std::uint32_t>(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i) {
        const auto& c = kept[i];
        auto& out = result->objects[i];
        out.x = c.x1 / transform.source_width;
        out.y = c.y1 / transform.source_height;
        out.w = (c.x2 - c.x1) / transform.source_width;
        out.h = (c.y2 - c.y1) / transform.source_height;
        out.class_id = c.class_id;
        out.score = c.score;
        std::snprintf(out.label, sizeof(out.label), "%s", label_for(profile, c.class_id));
        out.nKeypoints = static_cast<std::uint32_t>(std::min<std::size_t>(MAX_KEYPOINTS, c.keypoints.size()));
        std::copy_n(c.keypoints.begin(), out.nKeypoints, out.keypoints);
    }
    return 0;
}

}  // namespace rkdl
