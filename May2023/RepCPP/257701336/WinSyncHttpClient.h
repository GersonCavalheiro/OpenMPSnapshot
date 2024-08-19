

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AwsStringStream.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/standard/StandardHttpResponse.h>

namespace Aws
{
namespace Client
{
struct ClientConfiguration;
} 

namespace Http
{

class WinConnectionPoolMgr;


class AWS_CORE_API WinSyncHttpClient : public HttpClient
{
public:
using Base = HttpClient;

virtual ~WinSyncHttpClient();


std::shared_ptr<HttpResponse> MakeRequest(HttpRequest& request,
Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const;


virtual const char* GetLogTag() const { return "WinSyncHttpClient"; }
protected:

void SetOpenHandle(void* handle);

void* GetOpenHandle() const { return m_openHandle; }

void SetConnectionPoolManager(WinConnectionPoolMgr* connectionMgr);

WinConnectionPoolMgr* GetConnectionPoolManager() const { return m_connectionPoolMgr; }

void* AllocateWindowsHttpRequest(const Aws::Http::HttpRequest& request, void* connection) const;

bool m_allowRedirects;
private:

virtual void* OpenRequest(const Aws::Http::HttpRequest& request, void* connection, const Aws::StringStream& ss) const = 0;
virtual void DoAddHeaders(void* hHttpRequest, Aws::String& headerStr) const = 0;
virtual uint64_t DoWriteData(void* hHttpRequest, char* streamBuffer, uint64_t bytesRead) const = 0;
virtual bool DoReceiveResponse(void* hHttpRequest) const = 0;
virtual bool DoQueryHeaders(void* hHttpRequest, std::shared_ptr<Aws::Http::Standard::StandardHttpResponse>& response, Aws::StringStream& ss, uint64_t& read) const = 0;
virtual bool DoSendRequest(void* hHttpRequest) const = 0;
virtual bool DoReadData(void* hHttpRequest, char* body, uint64_t size, uint64_t& read) const = 0;
virtual void* GetClientModule() const = 0;

bool StreamPayloadToRequest(const HttpRequest& request, void* hHttpRequest, Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter) const;
void LogRequestInternalFailure() const;
std::shared_ptr<HttpResponse> BuildSuccessResponse(const Aws::Http::HttpRequest& request, void* hHttpRequest, Aws::Utils::RateLimits::RateLimiterInterface* readLimiter) const;
void AddHeadersToRequest(const HttpRequest& request, void* hHttpRequest) const;

void* m_openHandle;
WinConnectionPoolMgr* m_connectionPoolMgr;
};

} 
} 

