

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/block/shared/dyn/Traits.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>

#    include <type_traits>

namespace alpaka
{
class BlockSharedMemDynUniformCudaHipBuiltIn
: public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynUniformCudaHipBuiltIn>
{
};

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

namespace trait
{
template<typename T>
struct GetDynSharedMem<T, BlockSharedMemDynUniformCudaHipBuiltIn>
{
__device__ static auto getMem(BlockSharedMemDynUniformCudaHipBuiltIn const&) -> T*
{
extern __shared__ float4 shMem[];
return reinterpret_cast<T*>(shMem);
}
};
} 

#    endif

} 

#endif
