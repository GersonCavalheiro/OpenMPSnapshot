

#pragma once


#include "paraverkernelexception.h"

namespace NoLoad
{

enum class TNoLoadErrorCode
{
undefined = 0,
wrongTraceBodyVersion,
LAST
};

class NoLoadException: public ParaverKernelException
{

public:
NoLoadException( TNoLoadErrorCode whichCode = TNoLoadErrorCode::undefined,
const char *whichAuxMessage = "",
const char *whichFile = nullptr,
TExceptionLine whichLine = 0 )
{
code = whichCode;
auxMessage = whichAuxMessage;
file = whichFile;
line = whichLine;
}

protected:

static std::string moduleMessage;

TNoLoadErrorCode code;

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

}


