
#pragma once

#include <string>
#include <iostream>


#include "testing/test_case.h"

namespace Kratos::Testing {




class KRATOS_API(KRATOS_CORE) DistributedTestCase: public TestCase
{
public:

DistributedTestCase() = delete;

DistributedTestCase(DistributedTestCase const& rOther) = delete;

DistributedTestCase(std::string const& Name);

~DistributedTestCase() override;


DistributedTestCase& operator=(DistributedTestCase const& rOther) = delete;


void Run() override;

void Profile() override;


bool IsEnabled() const override;

bool IsDisabled() const override;


std::string Info() const override;

private:

void CheckRemoteFailure();

};

}
