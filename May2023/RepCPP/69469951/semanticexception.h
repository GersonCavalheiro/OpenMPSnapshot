


#pragma once


#include "paraverkernelexception.h"

enum class TSemanticErrorCode
{
undefined = 0,
maxParamExceeded,
LAST
};

class SemanticException: public ParaverKernelException
{
public:
SemanticException( TSemanticErrorCode whichCode = TSemanticErrorCode::undefined,
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

TSemanticErrorCode code;

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



