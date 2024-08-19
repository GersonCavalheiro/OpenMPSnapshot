

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/logging/FormattedLogSystem.h>

namespace Aws
{
namespace Utils
{
namespace Logging
{

class AWS_CORE_API LogcatLogSystem : public FormattedLogSystem
{
public:

using Base = FormattedLogSystem;

LogcatLogSystem(LogLevel logLevel) :
Base(logLevel)
{}

virtual ~LogcatLogSystem() {}

protected:

virtual void ProcessFormattedStatement(Aws::String&& statement) override;
};

} 
} 
} 


