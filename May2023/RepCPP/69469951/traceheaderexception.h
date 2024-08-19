


#pragma once


#include "paraverkernelexception.h"

enum class TTraceHeaderErrorCode
{
undefined = 0,
invalidApplNumber,
invalidTaskNumber,
invalidThreadNumber,
invalidNodeNumber,
invalidCPUNumber,
invalidTime,
invalidCommNumber,
unknownCommLine,
emptyBody,
LAST
};

class TraceHeaderException: public ParaverKernelException
{
public:
TraceHeaderException( TTraceHeaderErrorCode whichCode = TTraceHeaderErrorCode::undefined,
const char *whichAuxMessage = "",
const char *whichFile = nullptr,
TExceptionLine whichLine = 0 ) noexcept
{
code = whichCode;
auxMessage = whichAuxMessage;
file = whichFile;
line = whichLine;
}

protected:

static std::string moduleMessage;

TTraceHeaderErrorCode code;

private:
static const char *errorMessage[];

virtual const char *specificErrorMessage() const override
{
return errorMessage[ static_cast< int >( code ) ];
}

virtual std::string& specificModuleMessage() const
{
return moduleMessage;
}
};


