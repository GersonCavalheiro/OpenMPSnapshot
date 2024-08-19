

#pragma once

#include <aws/core/Core_EXPORTS.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/logging/LogLevel.h>

#include <atomic>

namespace Aws
{
namespace Utils
{
namespace Logging
{

class AWS_CORE_API FormattedLogSystem : public LogSystemInterface
{
public:
using Base = LogSystemInterface;


FormattedLogSystem(LogLevel logLevel);
virtual ~FormattedLogSystem() = default;


virtual LogLevel GetLogLevel(void) const override { return m_logLevel; }

void SetLogLevel(LogLevel logLevel) { m_logLevel.store(logLevel); }


virtual void Log(LogLevel logLevel, const char* tag, const char* formatStr, ...) override;


virtual void LogStream(LogLevel logLevel, const char* tag, const Aws::OStringStream &messageStream) override;

protected:

virtual void ProcessFormattedStatement(Aws::String&& statement) = 0;

private:
std::atomic<LogLevel> m_logLevel;
};

} 
} 
} 
