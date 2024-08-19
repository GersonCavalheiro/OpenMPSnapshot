

#pragma once

#include <aws/core/Core_EXPORTS.h>

namespace Aws
{
namespace Client
{

enum class CoreErrors;
template<typename ERROR_TYPE>
class AWSError;


class AWS_CORE_API RetryStrategy
{
public:
virtual ~RetryStrategy() {}

virtual bool ShouldRetry(const AWSError<CoreErrors>& error, long attemptedRetries) const = 0;


virtual long CalculateDelayBeforeNextRetry(const AWSError<CoreErrors>& error, long attemptedRetries) const = 0;

};

} 
} 
