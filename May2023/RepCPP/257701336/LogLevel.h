

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSString.h>

namespace Aws
{
namespace Utils
{
namespace Logging
{


enum class LogLevel : int
{
Off = 0,
Fatal = 1,
Error = 2,
Warn = 3,
Info = 4,
Debug = 5,
Trace = 6
};

AWS_CORE_API Aws::String GetLogLevelName(LogLevel logLevel);

} 
} 
} 
