


#pragma once

#include <hydra/detail/external/hydra_thrust/system/cuda/error.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/guarded_cuda_runtime_api.h>

namespace hydra_thrust
{

namespace system
{


error_code make_error_code(cuda::errc::errc_t e)
{
return error_code(static_cast<int>(e), cuda_category());
} 


error_condition make_error_condition(cuda::errc::errc_t e)
{
return error_condition(static_cast<int>(e), cuda_category());
} 


namespace cuda_cub
{

namespace detail
{


class cuda_error_category
: public error_category
{
public:
inline cuda_error_category(void) {}

inline virtual const char *name(void) const
{
return "cuda";
}

inline virtual std::string message(int ev) const
{
char const* const unknown_str  = "unknown error";
char const* const unknown_name = "cudaErrorUnknown";
char const* c_str  = ::cudaGetErrorString(static_cast<cudaError_t>(ev));
char const* c_name = ::cudaGetErrorName(static_cast<cudaError_t>(ev));
return std::string(c_name ? c_name : unknown_name)
+ ": " + (c_str ? c_str : unknown_str);
}

inline virtual error_condition default_error_condition(int ev) const
{
using namespace cuda::errc;

if(ev < ::cudaErrorApiFailureBase)
{
return make_error_condition(static_cast<errc_t>(ev));
}

return system_category().default_error_condition(ev);
}
}; 

} 

} 


const error_category &cuda_category(void)
{
static const hydra_thrust::system::cuda_cub::detail::cuda_error_category result;
return result;
}


} 

} 

