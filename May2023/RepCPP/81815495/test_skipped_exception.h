
#pragma once

#include <stdexcept>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>


#include "includes/exception.h"

namespace Kratos
{


class KRATOS_API(KRATOS_CORE) TestSkippedException : public Exception
{
public:


TestSkippedException();

explicit TestSkippedException(const std::string &rWhat);

TestSkippedException(const std::string &rWhat, const CodeLocation &Location);

TestSkippedException(TestSkippedException const &Other);

~TestSkippedException() noexcept override;


TestSkippedException& operator << (CodeLocation const& TheLocation);

template<class StreamValueType>
TestSkippedException& operator << (StreamValueType const& rValue)
{
Exception::operator << (rValue);
return *this;
}

TestSkippedException& operator << (std::ostream& (*pf)(std::ostream&));

TestSkippedException& operator << (const char * rString);


std::string Info() const override;

void PrintInfo(std::ostream &rOStream) const override;

void PrintData(std::ostream &rOStream) const override;


}; 



#define KRATOS_SKIP_TEST throw TestSkippedException("Test Skipped: ", KRATOS_CODE_LOCATION)

#define KRATOS_SKIP_TEST_IF(conditional) \
if (conditional)                 \
throw TestSkippedException("Test Skipped: ", KRATOS_CODE_LOCATION)

#define KRATOS_SKIP_TEST_IF_NOT(conditional) \
if (!(conditional))                  \
throw TestSkippedException("Test Skipped: ", KRATOS_CODE_LOCATION)


std::istream &operator>>(std::istream &rIStream,
TestSkippedException &rThis);

KRATOS_API(KRATOS_CORE)
std::ostream &operator<<(std::ostream &rOStream, const TestSkippedException &rThis);



} 
