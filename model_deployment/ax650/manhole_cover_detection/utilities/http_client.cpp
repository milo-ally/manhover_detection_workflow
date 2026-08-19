#include "http_client.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/time.h>
#include <errno.h>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <mutex>

HttpClient::Response HttpClient::post(const std::string& url, const std::string& jsonData, int timeoutSeconds) {
    Response response;
    response.statusCode = -1;
    
    // 解析 URL
    std::string host, path;
    int port;
    if (!parseUrl(url, host, port, path)) {
        response.error = "Invalid URL: " + url;
        return response;
    }
    
    // 連接 socket
    int sockfd = connectSocket(host, port, timeoutSeconds);
    if (sockfd < 0) {
        response.error = "Failed to connect to " + host + ":" + std::to_string(port);
        return response;
    }
    
    // 構建 HTTP 請求
    std::string request = buildRequest("POST", path, host, jsonData);
    
    // 發送請求
    ssize_t sent = send(sockfd, request.c_str(), request.length(), 0);
    if (sent < 0) {
        response.error = "Failed to send request";
        close(sockfd);
        return response;
    }
    
    // 接收響應
    response = parseResponse(sockfd);
    
    close(sockfd);
    return response;
}

bool HttpClient::parseUrl(const std::string& url, std::string& host, int& port, std::string& path) {
    // 簡單的 URL 解析：http://host:port/path
    size_t protocolEnd = url.find("://");
    if (protocolEnd == std::string::npos) {
        return false;
    }
    
    std::string protocol = url.substr(0, protocolEnd);
    if (protocol != "http") {
        return false;
    }
    
    size_t hostStart = protocolEnd + 3;
    size_t pathStart = url.find('/', hostStart);
    size_t portStart = url.find(':', hostStart);
    
    if (pathStart == std::string::npos) {
        path = "/";
        pathStart = url.length();
    } else {
        path = url.substr(pathStart);
    }
    
    if (portStart != std::string::npos && portStart < pathStart) {
        host = url.substr(hostStart, portStart - hostStart);
        std::string portStr = url.substr(portStart + 1, pathStart - portStart - 1);
        port = std::stoi(portStr);
    } else {
        host = url.substr(hostStart, pathStart - hostStart);
        port = 80;  // 默認 HTTP 端口
    }
    
    return true;
}

int HttpClient::connectSocket(const std::string& host, int port, int timeoutSeconds) {
    // 使用 getaddrinfo 代替 gethostbyname（線程安全）
    struct addrinfo hints, *result, *rp;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;  // 只使用 IPv4
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    hints.ai_protocol = 0;
    
    std::string portStr = std::to_string(port);
    int ret = getaddrinfo(host.c_str(), portStr.c_str(), &hints, &result);
    if (ret != 0) {
        return -1;  // DNS 解析失敗
    }
    
    // 嘗試連接每個地址
    int sockfd = -1;
    for (rp = result; rp != nullptr; rp = rp->ai_next) {
        // 創建 socket
        sockfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (sockfd < 0) {
            continue;  // 嘗試下一個地址
        }
        
        // 設置非阻塞模式（用於超時控制）
        int flags = fcntl(sockfd, F_GETFL, 0);
        if (flags < 0) {
            close(sockfd);
            sockfd = -1;
            continue;
        }
        if (fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) < 0) {
            close(sockfd);
            sockfd = -1;
            continue;
        }
        
        // 嘗試連接
        int connectResult = connect(sockfd, rp->ai_addr, rp->ai_addrlen);
        if (connectResult < 0) {
            if (errno == EINPROGRESS) {
                // 使用 select 實現超時
                fd_set writefds;
                struct timeval timeout;
                FD_ZERO(&writefds);
                FD_SET(sockfd, &writefds);
                timeout.tv_sec = timeoutSeconds;
                timeout.tv_usec = 0;
                
                int selectResult = select(sockfd + 1, nullptr, &writefds, nullptr, &timeout);
                if (selectResult <= 0) {
                    close(sockfd);
                    sockfd = -1;
                    continue;  // 超時或錯誤，嘗試下一個地址
                }
                
                // 檢查連接是否成功
                int error = 0;
                socklen_t len = sizeof(error);
                if (getsockopt(sockfd, SOL_SOCKET, SO_ERROR, &error, &len) < 0 || error != 0) {
                    close(sockfd);
                    sockfd = -1;
                    continue;  // 連接失敗，嘗試下一個地址
                }
            } else {
                close(sockfd);
                sockfd = -1;
                continue;  // 連接失敗，嘗試下一個地址
            }
        }
        
        // 連接成功，恢復阻塞模式
        fcntl(sockfd, F_SETFL, flags);
        break;  // 成功連接，退出循環
    }
    
    freeaddrinfo(result);  // 釋放 addrinfo 結構
    
    return sockfd;  // 返回 socket 描述符，如果所有地址都失敗則返回 -1
}

std::string HttpClient::buildRequest(const std::string& method, const std::string& path, 
                                     const std::string& host, const std::string& body) {
    std::ostringstream request;
    request << method << " " << path << " HTTP/1.1\r\n";
    request << "Host: " << host << "\r\n";
    request << "Content-Type: application/json\r\n";
    request << "Content-Length: " << body.length() << "\r\n";
    request << "Connection: close\r\n";
    request << "\r\n";
    request << body;
    
    return request.str();
}

HttpClient::Response HttpClient::parseResponse(int sockfd) {
    Response response;
    char buffer[4096];
    std::string fullResponse;
    
    // 接收響應（簡單實現，不處理分塊傳輸）
    // 設置接收超時，避免無限等待
    struct timeval timeout;
    timeout.tv_sec = 5;
    timeout.tv_usec = 0;
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    
    while (true) {
        ssize_t received = recv(sockfd, buffer, sizeof(buffer) - 1, 0);
        if (received <= 0) {
            break;
        }
        // 確保不會溢出（雖然 recv 已經限制了大小）
        if (received >= static_cast<ssize_t>(sizeof(buffer))) {
            received = sizeof(buffer) - 1;
        }
        buffer[received] = '\0';
        fullResponse.append(buffer, received);  // 使用 append 而不是 +=，更安全
        
        // 簡單檢查：如果收到完整的 HTTP 響應頭和部分 body，就停止
        // 限制最大響應大小，避免內存溢出
        if (fullResponse.length() > 100000) {  // 限制為 100KB
            break;
        }
        if (fullResponse.find("\r\n\r\n") != std::string::npos && 
            fullResponse.length() > 1000) {  // 假設響應不會太大
            break;
        }
    }
    
    if (fullResponse.empty()) {
        response.error = "No response received";
        return response;
    }
    
    // 解析狀態碼
    size_t firstLineEnd = fullResponse.find("\r\n");
    if (firstLineEnd != std::string::npos) {
        std::string firstLine = fullResponse.substr(0, firstLineEnd);
        size_t firstSpace = firstLine.find(' ');
        size_t secondSpace = firstLine.find(' ', firstSpace + 1);
        if (firstSpace != std::string::npos && secondSpace != std::string::npos) {
            std::string statusStr = firstLine.substr(firstSpace + 1, secondSpace - firstSpace - 1);
            response.statusCode = std::stoi(statusStr);
        }
    }
    
    // 提取響應體
    size_t headerEnd = fullResponse.find("\r\n\r\n");
    if (headerEnd != std::string::npos) {
        response.body = fullResponse.substr(headerEnd + 4);
    } else {
        response.body = fullResponse;
    }
    
    return response;
}

