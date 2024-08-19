

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
class WinINetConnectionPoolMgr;


class AWS_CORE_API WinINetSyncHttpClient : public WinSyncHttpClient
{
public:
using Base = WinSyncHttpClient;


WinINetSyncHttpClient(const Aws::Client::ClientConfiguration& clientConfig);
~WinINetSyncHttpClient();


const char* GetLogTag() const override { return "WinInetSyncHttpClient"; }
private:

void* OpenRequest(const Aws::Http::HttpRequest& request, void* connection, const Aws::StringStream& ss) const override;
void DoAddHeaders(void* hHttpRequest, Aws::String& headerStr) const override;
uint64_t DoWriteData(void* hHttpRequest, char* streamBuffer, uint64_t bytesRead) const override;
bool DoReceiveResponse(void* hHttpRequest) const override;
bool DoQueryHeaders(void* hHttpRequest, std::shared_ptr<Aws::Http::Standard::StandardHttpResponse>& response, Aws::StringStream& ss, uint64_t& read) const override;
bool DoSendRequest(void* hHttpRequest) const override;
bool DoReadData(void* hHttpRequest, char* body, uint64_t size, uint64_t& read) const override;
void* GetClientModule() const override;

WinINetSyncHttpClient &operator =(const WinINetSyncHttpClient &rhs);

bool m_usingProxy;
Aws::String m_proxyUserName;
Aws::String m_proxyPassword;
};

} 
} 

