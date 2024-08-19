


#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/curl/CurlHandleContainer.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <atomic>

namespace Aws
{
namespace Http
{


class AWS_CORE_API CurlHttpClient: public HttpClient
{
public:

using Base = HttpClient;

CurlHttpClient(const Aws::Client::ClientConfiguration& clientConfig);
std::shared_ptr<HttpResponse> MakeRequest(HttpRequest& request, Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const;

static void InitGlobalState();
static void CleanupGlobalState();

private:
mutable CurlHandleContainer m_curlHandleContainer;
bool m_isUsingProxy;
Aws::String m_proxyUserName;
Aws::String m_proxyPassword;
Aws::String m_proxyScheme;
Aws::String m_proxyHost;
unsigned m_proxyPort;
bool m_verifySSL;
Aws::String m_caPath;
Aws::String m_caFile;
bool m_allowRedirects;

static std::atomic<bool> isInit;

static size_t ReadBody(char* ptr, size_t size, size_t nmemb, void* userdata);
static size_t WriteData(char* ptr, size_t size, size_t nmemb, void* userdata);
static size_t WriteHeader(char* ptr, size_t size, size_t nmemb, void* userdata);

};

using PlatformHttpClient = CurlHttpClient;

} 
} 

