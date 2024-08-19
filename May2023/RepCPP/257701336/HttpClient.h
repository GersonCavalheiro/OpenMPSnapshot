

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace Aws
{
namespace Utils
{
namespace RateLimits
{
class RateLimiterInterface;
} 
} 

namespace Http
{
class HttpRequest;
class HttpResponse;


class AWS_CORE_API HttpClient
{
public:
HttpClient();
virtual ~HttpClient() {}


virtual std::shared_ptr<HttpResponse> MakeRequest(HttpRequest& request,
Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const = 0;


void DisableRequestProcessing();

void EnableRequestProcessing();

bool IsRequestProcessingEnabled() const;

void RetryRequestSleep(std::chrono::milliseconds sleepTime);

bool ContinueRequest(const Aws::Http::HttpRequest&) const;

private:

std::atomic< bool > m_disableRequestProcessing;

std::mutex m_requestProcessingSignalLock;
std::condition_variable m_requestProcessingSignal;
};

} 
} 


