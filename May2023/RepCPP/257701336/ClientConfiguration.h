

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/http/Scheme.h>
#include <aws/core/Region.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/http/HttpTypes.h>
#include <memory>

namespace Aws
{
namespace Utils
{
namespace Threading
{
class Executor;
} 

namespace RateLimits
{
class RateLimiterInterface;
} 
} 

namespace Client
{
class RetryStrategy; 


struct AWS_CORE_API ClientConfiguration
{
ClientConfiguration();

Aws::String userAgent;

Aws::Http::Scheme scheme;

Aws::String region;

bool useDualStack;

unsigned maxConnections;

long requestTimeoutMs;

long connectTimeoutMs;

std::shared_ptr<RetryStrategy> retryStrategy;

Aws::String endpointOverride;

Aws::Http::Scheme proxyScheme;

Aws::String proxyHost;

unsigned proxyPort;

Aws::String proxyUserName;

Aws::String proxyPassword;

std::shared_ptr<Aws::Utils::Threading::Executor> executor;

bool verifySSL;

Aws::String caPath;

Aws::String caFile;

std::shared_ptr<Aws::Utils::RateLimits::RateLimiterInterface> writeRateLimiter;

std::shared_ptr<Aws::Utils::RateLimits::RateLimiterInterface> readRateLimiter;

Aws::Http::TransferLibType httpLibOverride;

bool followRedirects;

bool enableClockSkewAdjustment;
};

} 
} 


