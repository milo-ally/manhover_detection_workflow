#ifndef ALARM_FILTER_H
#define ALARM_FILTER_H

#include "../../include/ai_interface.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>

/**
 * 告警過濾器：決定哪些檢測結果需要上報告警
 * 支持通過配置靈活定義告警規則，無需修改代碼
 */
class AlarmFilter {
public:
    /**
     * 告警規則配置
     */
    struct AlarmRule {
        std::string model_type;           // 模型類型：如 "helmet", "fall", "fire"
        std::string alarm_type;           // 告警類型：如 "未戴安全帽", "人員跌倒", "煙火檢測"
        std::string severity;             // 嚴重程度：low, medium, high, critical
        std::vector<std::string> labels;  // 需要告警的標籤列表（例如：["no-helmet"]）
        bool report_all = false;          // 如果為 true，上報所有檢測到的目標
    };
    
    /**
     * 初始化告警規則
     * @param rules 告警規則列表
     */
    static void initRules(const std::vector<AlarmRule>& rules);
    
    /**
     * 從配置文件加載告警規則（JSON格式）
     * @param configPath 配置文件路徑
     * @return 是否成功加載
     */
    static bool loadRulesFromFile(const std::string& configPath);
    
    /**
     * 根據模型名稱和檢測結果，判斷是否需要告警並返回告警信息
     * @param modelName 模型名稱
     * @param result 檢測結果
     * @param alarmType 輸出：告警類型（如果返回 true）
     * @param modelType 輸出：模型類型（如果返回 true）
     * @param severity 輸出：嚴重程度（如果返回 true）
     * @param alarmObjects 輸出：需要告警的目標列表（如果返回 true）
     * @return 是否需要告警
     */
    static bool shouldReportAlarm(
        const std::string& modelName,
        const AI_RESULT_T& result,
        std::string& alarmType,
        std::string& modelType,
        std::string& severity,
        std::vector<AI_OBJ_T>& alarmObjects
    );
    
    /**
     * 添加或更新告警規則
     * @param rule 告警規則
     */
    static void addRule(const AlarmRule& rule);
    
    /**
     * 清除所有規則
     */
    static void clearRules();
    
private:
    static std::map<std::string, AlarmRule> rules_;  // 模型類型 -> 告警規則
    static std::mutex rulesMutex_;
    
    /**
     * 根據模型名稱匹配規則
     * @param modelName 模型名稱
     * @return 匹配的規則，如果沒有匹配則返回 nullptr
     */
    static const AlarmRule* findRule(const std::string& modelName);
};

#endif // ALARM_FILTER_H

