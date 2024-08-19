

#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/CudaHipCommon.hpp>

#    include <iostream>
#    include <stdexcept>
#    include <string>

namespace alpaka::cuda::detail
{
ALPAKA_FN_HOST inline auto cudaDrvCheck(CUresult const& error, char const* desc, char const* file, int const& line)
-> void
{
if(error == CUDA_SUCCESS)
return;

char const* cu_err_name = nullptr;
char const* cu_err_string = nullptr;
CUresult cu_result_name = cuGetErrorName(error, &cu_err_name);
CUresult cu_result_string = cuGetErrorString(error, &cu_err_string);
std::string sError = std::string(file) + "(" + std::to_string(line) + ") " + std::string(desc) + " : '";
if(cu_result_name == CUDA_SUCCESS && cu_result_string == CUDA_SUCCESS)
{
sError += std::string(cu_err_name) + "': '" + std::string(cu_err_string) + "'!";
}
else
{
if(cu_result_name == CUDA_ERROR_INVALID_VALUE)
{
sError += " cuGetErrorName: 'Invalid Value'!";
}
if(cu_result_string == CUDA_ERROR_INVALID_VALUE)
{
sError += " cuGetErrorString: 'Invalid Value'!";
}
}
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
std::cerr << sError << std::endl;
#    endif
ALPAKA_DEBUG_BREAK;
throw std::runtime_error(sError);
}
} 

#    define ALPAKA_CUDA_DRV_CHECK(cmd) ::alpaka::cuda::detail::cudaDrvCheck(cmd, #    cmd, __FILE__, __LINE__)

#    include <alpaka/core/UniformCudaHip.hpp>

#endif
