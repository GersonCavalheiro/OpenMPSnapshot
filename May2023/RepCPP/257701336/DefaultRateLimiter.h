

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/ratelimiter/RateLimiterInterface.h>
#include <aws/core/utils/memory/stl/AWSFunction.h>

#include <algorithm>
#include <mutex>
#include <thread>

namespace Aws
{
namespace Utils
{
namespace RateLimits
{

template<typename CLOCK = std::chrono::high_resolution_clock, typename DUR = std::chrono::seconds, bool RENORMALIZE_RATE_CHANGES = true>
class DefaultRateLimiter : public RateLimiterInterface
{
public:
using Base = RateLimiterInterface;

using InternalTimePointType = std::chrono::time_point<CLOCK>;
using ElapsedTimeFunctionType = std::function< InternalTimePointType() >;


DefaultRateLimiter(int64_t maxRate, ElapsedTimeFunctionType elapsedTimeFunction = AWS_BUILD_FUNCTION(CLOCK::now)) :
m_elapsedTimeFunction(elapsedTimeFunction),
m_maxRate(0),
m_accumulatorLock(),
m_accumulator(0),
m_accumulatorFraction(0),
m_accumulatorUpdated(),
m_replenishNumerator(0),
m_replenishDenominator(0),
m_delayNumerator(0),
m_delayDenominator(0)
{
static_assert(DUR::period::num > 0, "Rate duration must have positive numerator");
static_assert(DUR::period::den > 0, "Rate duration must have positive denominator");
static_assert(CLOCK::duration::period::num > 0, "RateLimiter clock duration must have positive numerator");
static_assert(CLOCK::duration::period::den > 0, "RateLimiter clock duration must have positive denominator");

SetRate(maxRate, true);
}

virtual ~DefaultRateLimiter() = default;


virtual DelayType ApplyCost(int64_t cost) override
{
std::lock_guard<std::recursive_mutex> lock(m_accumulatorLock);

auto now = m_elapsedTimeFunction();
auto elapsedTime = (now - m_accumulatorUpdated).count();

auto temp = elapsedTime * m_replenishNumerator + m_accumulatorFraction;
m_accumulator += temp / m_replenishDenominator;
m_accumulatorFraction = temp % m_replenishDenominator;

m_accumulator = (std::min)(m_accumulator, m_maxRate);
if (m_accumulator == m_maxRate)
{
m_accumulatorFraction = 0;
}

DelayType delay(0);
if (m_accumulator < 0)
{
delay = DelayType(-m_accumulator * m_delayDenominator / m_delayNumerator);
}

m_accumulator -= cost;
m_accumulatorUpdated = now;

return delay;
}


virtual void ApplyAndPayForCost(int64_t cost) override
{
auto costInMilliseconds = ApplyCost(cost);
if(costInMilliseconds.count() > 0)
{
std::this_thread::sleep_for(costInMilliseconds);
}
}


virtual void SetRate(int64_t rate, bool resetAccumulator = false) override
{
std::lock_guard<std::recursive_mutex> lock(m_accumulatorLock);

rate = (std::max)(static_cast<int64_t>(1), rate);

if (resetAccumulator)
{
m_accumulator = rate;
m_accumulatorFraction = 0;
m_accumulatorUpdated = m_elapsedTimeFunction();
}
else
{
ApplyCost(0); 

if (ShouldRenormalizeAccumulatorOnRateChange())
{
m_accumulator = m_accumulator * rate / m_maxRate;
m_accumulatorFraction = m_accumulatorFraction * rate / m_maxRate;
}
}

m_maxRate = rate;

m_replenishNumerator = m_maxRate * DUR::period::den * CLOCK::duration::period::num;
m_replenishDenominator = DUR::period::num * CLOCK::duration::period::den;
auto gcd = ComputeGCD(m_replenishNumerator, m_replenishDenominator);
m_replenishNumerator /= gcd;
m_replenishDenominator /= gcd;

m_delayNumerator = m_maxRate * DelayType::period::num * DUR::period::den;
m_delayDenominator = DelayType::period::den * DUR::period::num;
gcd = ComputeGCD(m_delayNumerator, m_delayDenominator);
m_delayNumerator /= gcd;
m_delayDenominator /= gcd;
}

private:

int64_t ComputeGCD(int64_t num1, int64_t num2) const
{
while (num2 != 0)
{
int64_t rem = num1 % num2;
num1 = num2;
num2 = rem;
}

return num1;
}

bool ShouldRenormalizeAccumulatorOnRateChange() const { return RENORMALIZE_RATE_CHANGES; }

ElapsedTimeFunctionType m_elapsedTimeFunction;

int64_t m_maxRate;

std::recursive_mutex m_accumulatorLock;

int64_t m_accumulator;

int64_t m_accumulatorFraction;

InternalTimePointType m_accumulatorUpdated;

int64_t m_replenishNumerator;
int64_t m_replenishDenominator;
int64_t m_delayNumerator;
int64_t m_delayDenominator;
};

} 
} 
} 
