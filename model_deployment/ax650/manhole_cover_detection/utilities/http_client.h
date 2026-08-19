#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

#include <string>
#include <vector>

/**
 * 簡單的 HTTP 客戶端（用於上報告警）
 * 使用 socket 實現，不依賴外部庫
 */
class HttpClient {
public:
    struct Response {
        int statusCode;
        std::string body;
        std::string error;
    };
    
    /**
     * 發送 HTTP POST 請求
     * @param url 完整的 URL（例如：http://192.168.1.100:8001/api/alarms）
     * @param jsonData JSON 格式的請求體
     * @param timeoutSeconds 超時時間（秒），默認 5 秒
     * @return Response 響應對象
     */
    static Response post(const std::string& url, const std::string& jsonData, int timeoutSeconds = 5);
    
private:
    static bool parseUrl(const std::string& url, std::string& host, int& port, std::string& path);
    static int connectSocket(const std::string& host, int port, int timeoutSeconds);
    static std::string buildRequest(const std::string& method, const std::string& path, 
                                    const std::string& host, const std::string& body);
    static Response parseResponse(int sockfd);
};

#endif // HTTP_CLIENT_H

