

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <memory>

namespace Aws
{
namespace Utils
{
namespace Logging
{
class LogSystemInterface;



AWS_CORE_API void InitializeAWSLogging(const std::shared_ptr<LogSystemInterface>& logSystem);


AWS_CORE_API void ShutdownAWSLogging(void);


AWS_CORE_API LogSystemInterface* GetLogSystem();



AWS_CORE_API void PushLogger(const std::shared_ptr<LogSystemInterface> &logSystem);


AWS_CORE_API void PopLogger();

} 
} 
} 
