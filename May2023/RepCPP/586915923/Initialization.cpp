

#include <cuda.h>
#include <TACUDA.h>

#include "common/Environment.hpp"

using namespace tacuda;

#pragma GCC visibility push(default)

extern "C" {

CUresult
tacudaInit(unsigned int flags)
{
CUresult eret = cuInit(flags);
if (eret == CUDA_SUCCESS) {
Environment::initialize();
}
return eret;
}

CUresult
tacudaFinalize()
{
Environment::finalize();
return CUDA_SUCCESS;
}

} 

#pragma GCC visibility pop
