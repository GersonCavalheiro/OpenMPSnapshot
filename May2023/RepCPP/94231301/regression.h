
#pragma once

#include "platform.h"

#include <vector>

namespace embree
{

struct RegressionTest 
{ 
RegressionTest (std::string name) : name(name) {}
virtual bool run() = 0;
std::string name;
};


void registerRegressionTest(RegressionTest* test);


RegressionTest* getRegressionTest(size_t index);
}
