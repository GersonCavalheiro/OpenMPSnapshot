

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <stdint.h>
#include <chrono>

namespace Aws
{
namespace Utils
{
namespace RateLimits
{

class RateLimiterInterface
{
public:
using DelayType = std::chrono::milliseconds;

virtual ~RateLimiterInterface() {}

virtual DelayType ApplyCost(int64_t cost) = 0;

virtual void ApplyAndPayForCost(int64_t cost) = 0;

virtual void SetRate(int64_t rate, bool resetAccumulator = false) = 0;
};

} 
} 
} 