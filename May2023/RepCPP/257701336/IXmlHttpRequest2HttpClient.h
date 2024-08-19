

#pragma once

#include <aws/core/http/HttpClient.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/ResourceManager.h>

#include <wrl.h> 

struct IXMLHTTPRequest2;

namespace Aws
{
namespace Http
{
typedef Microsoft::WRL::ComPtr<IXMLHTTPRequest2> HttpRequestComHandle;


class AWS_CORE_API IXmlHttpRequest2HttpClient : public HttpClient
{
public:

IXmlHttpRequest2HttpClient(const Aws::Client::ClientConfiguration& clientConfiguration);
virtual ~IXmlHttpRequest2HttpClient();


virtual std::shared_ptr<HttpResponse> MakeRequest(HttpRequest& request,
Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const;


static void InitCOM();

private:
void FillClientSettings(const HttpRequestComHandle&) const;

mutable Aws::Utils::ExclusiveOwnershipResourceManager<HttpRequestComHandle> m_resourceManager;
Aws::String m_proxyUserName;
Aws::String m_proxyPassword;
size_t m_poolSize;
bool m_followRedirects;
bool m_verifySSL;
size_t m_totalTimeoutMs;
};
}
}

