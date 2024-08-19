


#pragma once


#include "paraverkernelexception.h"

enum class THistogramErrorCode
{
undefined = 0,
noControlWindow,
LAST
};

class HistogramException: public ParaverKernelException
{
public:
HistogramException( THistogramErrorCode whichCode = THistogramErrorCode::undefined,
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
THistogramErrorCode code;

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


