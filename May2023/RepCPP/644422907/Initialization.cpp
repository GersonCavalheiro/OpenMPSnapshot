

#include <TAHIP.h>

#include "common/Environment.hpp"

using namespace tahip;

#pragma GCC visibility push(default)

extern "C" {

hipError_t
tahipInit(unsigned int flags)
{
hipError_t eret = hipInit(flags);
if (eret == hipSuccess) {
Environment::initialize();
}
return eret;
}

hipError_t
tahipFinalize()
{
Environment::finalize();
return hipSuccess;
}

} 

#pragma GCC visibility pop
