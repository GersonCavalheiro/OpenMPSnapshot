

#include <algorithm>




#include "includes/define.h"
#include "input_output/logger.h"


namespace Kratos
{

Logger::Logger(std::string const& TheLabel) : mCurrentMessage(TheLabel)
{
}

Logger::~Logger()
{
auto outputs = GetOutputsInstance();
#pragma omp critical
{
GetDefaultOutputInstance().WriteMessage(mCurrentMessage);
for (auto i_output = outputs.begin(); i_output != outputs.end(); ++i_output)
(*i_output)->WriteMessage(mCurrentMessage);
}
}

void Logger::AddOutput(LoggerOutput::Pointer pTheOutput)
{
#pragma omp critical
{
GetOutputsInstance().push_back(pTheOutput);
}
}

void Logger::RemoveOutput(LoggerOutput::Pointer pTheOutput)
{
KRATOS_TRY

#pragma omp critical
{
auto i = std::find(GetOutputsInstance().begin(), GetOutputsInstance().end(), pTheOutput);
if (i != GetOutputsInstance().end()) {
GetOutputsInstance().erase(i);
}  
}

KRATOS_CATCH("");
}

void Logger::Flush() {
auto outputs = GetOutputsInstance();
GetDefaultOutputInstance().Flush();
for (auto i_output = outputs.begin(); i_output != outputs.end(); ++i_output) {
(*i_output)->Flush();
}
}

std::string Logger::Info() const
{
return "Logger";
}

void Logger::PrintInfo(std::ostream& rOStream) const
{
}
void Logger::PrintData(std::ostream& rOStream) const
{
}

Logger& Logger::operator << (std::ostream& (*pf)(std::ostream&))
{
mCurrentMessage << pf;

return *this;
}

Logger& Logger::operator << (const char * rString)
{
mCurrentMessage << rString;

return *this;
}

Logger& Logger::operator << (CodeLocation const& TheLocation)
{
mCurrentMessage << TheLocation;

return *this;
}

Logger& Logger::operator << (Severity const& TheSeverity)
{
mCurrentMessage << TheSeverity;

return *this;
}

Logger& Logger::operator << (Category const& TheCategory)
{
mCurrentMessage << TheCategory;

return *this;
}



}  
