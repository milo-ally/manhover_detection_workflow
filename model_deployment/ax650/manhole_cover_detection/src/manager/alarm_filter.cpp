#include "alarm_filter.h"
#include "../../utilities/json.hpp"
#include "../../utilities/sample_log.h"
#include <fstream>
#include <mutex>
#include <algorithm>
#include <cstring>

std::map<std::string, AlarmFilter::AlarmRule> AlarmFilter::rules_;
std::mutex AlarmFilter::rulesMutex_;

void AlarmFilter::initRules(const std::vector<AlarmRule>& rules) {
    std::lock_guard<std::mutex> lock(rulesMutex_);
    rules_.clear();
    for (const auto& rule : rules) {
        rules_[rule.model_type] = rule;
    }
    ALOGN("[AlarmFilter] 初始化了 %zu 個告警規則", rules.size());
}

bool AlarmFilter::loadRulesFromFile(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        ALOGW("[AlarmFilter] 無法打開配置文件: %s", configPath.c_str());
        return false;
    }
    
    try {
        nlohmann::json config;
        file >> config;
        
        std::vector<AlarmRule> rules;
        
        if (config.contains("alarm_rules") && config["alarm_rules"].is_array()) {
            for (const auto& ruleJson : config["alarm_rules"]) {
                AlarmRule rule;
                rule.model_type = ruleJson.value("model_type", "");
                rule.alarm_type = ruleJson.value("alarm_type", "");
                rule.severity = ruleJson.value("severity", "high");
                rule.report_all = ruleJson.value("report_all", false);
                
                if (ruleJson.contains("labels") && ruleJson["labels"].is_array()) {
                    for (const auto& label : ruleJson["labels"]) {
                        rule.labels.push_back(label.get<std::string>());
                    }
                }
                
                if (!rule.model_type.empty() && !rule.alarm_type.empty()) {
                    rules.push_back(rule);
                    ALOGN("[AlarmFilter] 加載規則: model_type=%s, alarm_type=%s, labels=%zu",
                          rule.model_type.c_str(), rule.alarm_type.c_str(), rule.labels.size());
                }
            }
        }
        
        initRules(rules);
        return true;
    } catch (const std::exception& e) {
        ALOGE("[AlarmFilter] 解析配置文件失敗: %s", e.what());
        return false;
    }
}

const AlarmFilter::AlarmRule* AlarmFilter::findRule(const std::string& modelName) {
    std::lock_guard<std::mutex> lock(rulesMutex_);
    
    // 嘗試精確匹配
    for (const auto& pair : rules_) {
        if (modelName.find(pair.first) != std::string::npos) {
            return &pair.second;
        }
    }
    
    return nullptr;
}

bool AlarmFilter::shouldReportAlarm(
    const std::string& modelName,
    const AI_RESULT_T& result,
    std::string& alarmType,
    std::string& modelType,
    std::string& severity,
    std::vector<AI_OBJ_T>& alarmObjects
) {
    alarmObjects.clear();
    
    // 如果沒有檢測到目標，不需要告警
    if (result.nObjSize == 0) {
        return false;
    }
    
    // 查找匹配的規則
    const AlarmRule* rule = findRule(modelName);
    if (!rule) {
        // 如果沒有找到規則，使用默認規則（向後兼容）
        // 首先檢查檢測結果中的標籤來推斷模型類型（即使模型名稱為空）
        bool hasNoHelmet = false;
        for (unsigned int i = 0; i < result.nObjSize; i++) {
            if (strcmp(result.objects[i].label, "no-helmet") == 0) {
                hasNoHelmet = true;
                break;
            }
        }
        
        // 如果檢測到 no-helmet，即使模型名稱為空也認為是安全帽模型
        if (hasNoHelmet || (!modelName.empty() && modelName.find("helmet") != std::string::npos)) {
            modelType = "helmet";
            alarmType = "未戴安全帽";
            severity = "high";
            for (unsigned int i = 0; i < result.nObjSize; i++) {
                if (strcmp(result.objects[i].label, "no-helmet") == 0) {
                    alarmObjects.push_back(result.objects[i]);
                }
            }
            return !alarmObjects.empty();
        } else if (!modelName.empty() && 
                   (modelName.find("fall") != std::string::npos || 
                    modelName.find("pose") != std::string::npos)) {
            modelType = "fall";
            alarmType = "人員跌倒";
            severity = "high";
            for (unsigned int i = 0; i < result.nObjSize; i++) {
                alarmObjects.push_back(result.objects[i]);
            }
            return !alarmObjects.empty();
        }
        return false;
    }
    
    // 使用規則過濾
    modelType = rule->model_type;
    alarmType = rule->alarm_type;
    severity = rule->severity;
    
    if (rule->report_all) {
        // 上報所有目標
        for (unsigned int i = 0; i < result.nObjSize; i++) {
            alarmObjects.push_back(result.objects[i]);
        }
    } else {
        // 只上報匹配標籤的目標
        for (unsigned int i = 0; i < result.nObjSize; i++) {
            for (const auto& label : rule->labels) {
                if (strcmp(result.objects[i].label, label.c_str()) == 0) {
                    alarmObjects.push_back(result.objects[i]);
                    break;  // 找到匹配就跳出內層循環
                }
            }
        }
    }
    
    return !alarmObjects.empty();
}

void AlarmFilter::addRule(const AlarmRule& rule) {
    std::lock_guard<std::mutex> lock(rulesMutex_);
    rules_[rule.model_type] = rule;
    ALOGN("[AlarmFilter] 添加告警規則: model_type=%s, alarm_type=%s",
          rule.model_type.c_str(), rule.alarm_type.c_str());
}

void AlarmFilter::clearRules() {
    std::lock_guard<std::mutex> lock(rulesMutex_);
    rules_.clear();
    ALOGN("[AlarmFilter] 清除所有告警規則");
}

