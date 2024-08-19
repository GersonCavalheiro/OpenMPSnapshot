

#pragma once


#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
#include <hydra/detail/external/hydra_thrust/system/cuda/config.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/transform.h>
#include <hydra/detail/external/hydra_thrust/functional.h>

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

namespace __copy {

template <class Derived,
class InputIt,
class OutputIt>
OutputIt HYDRA_THRUST_RUNTIME_FUNCTION
device_to_device(execution_policy<Derived>& policy,
InputIt                    first,
InputIt                    last,
OutputIt                   result)
{
typedef typename hydra_thrust::iterator_traits<InputIt>::value_type InputTy;
return cuda_cub::transform(policy,
first,
last,
result,
hydra_thrust::identity<InputTy>());
}

}    

}    
HYDRA_THRUST_END_NS
#endif
