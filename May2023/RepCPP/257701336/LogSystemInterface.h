

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

namespace Aws
{
namespace Utils
{
namespace Logging
{
enum class LogLevel : int;


class AWS_CORE_API LogSystemInterface
{
public:
virtual ~LogSystemInterface() = default;


virtual LogLevel GetLogLevel(void) const = 0;

virtual void Log(LogLevel logLevel, const char* tag, const char* formatStr, ...) = 0;

virtual void LogStream(LogLevel logLevel, const char* tag, const Aws::OStringStream &messageStream) = 0;

};

} 
} 
} 
