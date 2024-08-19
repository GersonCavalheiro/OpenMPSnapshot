


#pragma once


#include <exception>
#include <iostream>
#include <string>

typedef int TExceptionLine;

enum class TErrorCode
{
undefined = 0,
emptytrace,
cannotOpenTrace,
nullOperand,
memoryError,
gzipNotSupported,
undefinedToolID,
undefinedToolName,
indexOutOfRange,
downloadFailed,
LAST
};

class ParaverKernelException : public std::exception
{

public:

static std::ostream& defaultPrintStream;

ParaverKernelException( TErrorCode whichCode = TErrorCode::undefined,
const char *whichAuxMessage = "",
const char *whichFile = nullptr,
TExceptionLine whichLine = 0 ) noexcept;

~ParaverKernelException() noexcept = default;

const char *what() const noexcept;

void printMessage( std::ostream& printStream = defaultPrintStream )
{
printStream << what();
}

protected:

static std::string kernelMessage;

TErrorCode code;

std::string auxMessage;

const char *file;

TExceptionLine line;

virtual const char *specificErrorMessage() const
{
return errorMessage[ static_cast<int>( code ) ];
}

private:
static const char *errorMessage[];
std::string message;

};


