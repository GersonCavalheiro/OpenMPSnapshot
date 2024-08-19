

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/HttpClient.h>
#include <aws/core/http/windows/WinSyncHttpClient.h>

namespace Aws
{
namespace Client
{
struct ClientConfiguration;
} 

namespace Http
{

class WinHttpConnectionPoolMgr;


class AWS_CORE_API WinHttpSyncHttpClient : public WinSyncHttpClient
{
public:
using Base = WinSyncHttpClient;


WinHttpSyncHttpClient(const Aws::Client::ClientConfiguration& clientConfig);
~WinHttpSyncHttpClient();


const char* GetLogTag() const override { return "WinHttpSyncHttpClient"; }

private:
void* OpenRequest(const Aws::Http::HttpRequest& request, void* connection, const Aws::StringStream& ss) const override;
void DoAddHeaders(void* httpRequest, Aws::String& headerStr) const override;
uint64_t DoWriteData(void* httpRequest, char* streamBuffer, uint64_t bytesRead) const override;
bool DoReceiveResponse(void* httpRequest) const override;
bool DoQueryHeaders(void* httpRequest, std::shared_ptr<Aws::Http::Standard::StandardHttpResponse>& response, Aws::StringStream& ss, uint64_t& read) const override;
bool DoSendRequest(void* httpRequest) const override;
bool DoReadData(void* hHttpRequest, char* body, uint64_t size, uint64_t& read) const override;
void* GetClientModule() const override;

bool m_usingProxy;
bool m_verifySSL;
Aws::WString m_proxyUserName;
Aws::WString m_proxyPassword;
};

} 
} 

