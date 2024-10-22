

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/warp/Traits.hpp>

#    include <cstdint>

namespace alpaka::warp
{
class WarpUniformCudaHipBuiltIn : public concepts::Implements<ConceptWarp, WarpUniformCudaHipBuiltIn>
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
template<>
struct GetSize<WarpUniformCudaHipBuiltIn>
{
__device__ static auto getSize(warp::WarpUniformCudaHipBuiltIn const& ) -> std::int32_t
{
return warpSize;
}
};

template<>
struct Activemask<WarpUniformCudaHipBuiltIn>
{
__device__ static auto activemask(warp::WarpUniformCudaHipBuiltIn const& )
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
-> std::uint32_t
#        else
-> std::uint64_t
#        endif
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return __activemask();
#        else
return __ballot(1);
#        endif
}
};

template<>
struct All<WarpUniformCudaHipBuiltIn>
{
__device__ static auto all(
[[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
std::int32_t predicate) -> std::int32_t
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return __all_sync(activemask(warp), predicate);
#        else
return __all(predicate);
#        endif
}
};

template<>
struct Any<WarpUniformCudaHipBuiltIn>
{
__device__ static auto any(
[[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
std::int32_t predicate) -> std::int32_t
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return __any_sync(activemask(warp), predicate);
#        else
return __any(predicate);
#        endif
}
};

template<>
struct Ballot<WarpUniformCudaHipBuiltIn>
{
__device__ static auto ballot(
[[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
std::int32_t predicate)
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
-> std::uint32_t
#        else
-> std::uint64_t
#        endif
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return __ballot_sync(activemask(warp), predicate);
#        else
return __ballot(predicate);
#        endif
}
};

template<>
struct Shfl<WarpUniformCudaHipBuiltIn>
{
__device__ static auto shfl(
[[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
float val,
int srcLane,
std::int32_t width) -> float
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return __shfl_sync(activemask(warp), val, srcLane, width);
#        else
return __shfl(val, srcLane, width);
#        endif
}
__device__ static auto shfl(
[[maybe_unused]] warp::WarpUniformCudaHipBuiltIn const& warp,
std::int32_t val,
int srcLane,
std::int32_t width) -> std::int32_t
{
#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
return __shfl_sync(activemask(warp), val, srcLane, width);
#        else
return __shfl(val, srcLane, width);
#        endif
}
};
} 
#    endif
} 

#endif
