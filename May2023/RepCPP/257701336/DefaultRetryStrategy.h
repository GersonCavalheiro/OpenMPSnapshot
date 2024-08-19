

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/client/RetryStrategy.h>

namespace Aws
{
namespace Client
{

class AWS_CORE_API DefaultRetryStrategy : public RetryStrategy
{
public:

DefaultRetryStrategy(long maxRetries = 10, long scaleFactor = 25) :
m_scaleFactor(scaleFactor), m_maxRetries(maxRetries)  
{}

bool ShouldRetry(const AWSError<CoreErrors>& error, long attemptedRetries) const;

long CalculateDelayBeforeNextRetry(const AWSError<CoreErrors>& error, long attemptedRetries) const;

private:
long m_scaleFactor;
long m_maxRetries;
};

} 
} 
