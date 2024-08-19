


#pragma once


#include "paraverkernelexception.h"

enum class TKWindowErrorCode
{
undefined = 0,
invalidLevel,
LAST
};

class KWindowException: public ParaverKernelException
{
public:
KWindowException( TKWindowErrorCode whichCode = TKWindowErrorCode::undefined,
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
TKWindowErrorCode code;

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



