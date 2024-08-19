

#pragma once

#include <aws/core/Core_EXPORTS.h>

#include <aws/core/utils/logging/FormattedLogSystem.h>

namespace Aws
{
namespace Utils
{
namespace Logging
{

class AWS_CORE_API ConsoleLogSystem : public FormattedLogSystem
{
public:

using Base = FormattedLogSystem;

ConsoleLogSystem(LogLevel logLevel) :
Base(logLevel)
{}

virtual ~ConsoleLogSystem() {}

protected:

virtual void ProcessFormattedStatement(Aws::String&& statement) override;
};

} 
} 
} 
