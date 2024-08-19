

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/http/HttpTypes.h>
#include <aws/core/utils/memory/stl/AWSStreamFwd.h>

namespace Aws
{
namespace Client
{
struct ClientConfiguration;
} 
namespace Http
{
class URI;
class HttpClient;
class HttpRequest;


class AWS_CORE_API HttpClientFactory
{
public:
virtual ~HttpClientFactory() {}


virtual std::shared_ptr<HttpClient> CreateHttpClient(const Aws::Client::ClientConfiguration& clientConfiguration) const = 0;

virtual std::shared_ptr<HttpRequest> CreateHttpRequest(const Aws::String& uri, HttpMethod method, const Aws::IOStreamFactory& streamFactory) const = 0;

virtual std::shared_ptr<HttpRequest> CreateHttpRequest(const URI& uri, HttpMethod method, const Aws::IOStreamFactory& streamFactory) const = 0;

virtual void InitStaticState() {}
virtual void CleanupStaticState() {}
};


AWS_CORE_API void SetInitCleanupCurlFlag(bool initCleanupFlag);
AWS_CORE_API void SetInstallSigPipeHandlerFlag(bool installHandler);
AWS_CORE_API void InitHttp();
AWS_CORE_API void CleanupHttp();
AWS_CORE_API void SetHttpClientFactory(const std::shared_ptr<HttpClientFactory>& factory);
AWS_CORE_API std::shared_ptr<HttpClient> CreateHttpClient(const Aws::Client::ClientConfiguration& clientConfiguration);
AWS_CORE_API std::shared_ptr<HttpRequest> CreateHttpRequest(const Aws::String& uri, HttpMethod method, const Aws::IOStreamFactory& streamFactory);
AWS_CORE_API std::shared_ptr<HttpRequest> CreateHttpRequest(const URI& uri, HttpMethod method, const Aws::IOStreamFactory& streamFactory);

} 
} 

