

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/rand/Traits.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>

#        include <curand_kernel.h>

#    elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <alpaka/core/Hip.hpp>

#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wduplicate-decl-specifier"
#        endif

#        if HIP_VERSION >= 50200000
#            include <hiprand/hiprand_kernel.h>
#        else
#            include <hiprand_kernel.h>
#        endif

#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

#    include <type_traits>

namespace alpaka::rand
{
template<typename TApi>
class RandUniformCudaHipRand : public concepts::Implements<ConceptRand, RandUniformCudaHipRand<TApi>>
{
};

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

namespace distribution::uniform_cuda_hip
{
template<typename T>
class NormalReal;

template<typename T>
class UniformReal;

template<typename T>
class UniformUint;
} 

namespace engine::uniform_cuda_hip
{
class Xor
{
public:
Xor() = default;

__device__ Xor(
std::uint32_t const& seed,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0)
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
curand_init(seed, subsequence, offset, &state);
#        else
hiprand_init(seed, subsequence, offset, &state);
#        endif
}

private:
template<typename T>
friend class distribution::uniform_cuda_hip::NormalReal;
template<typename T>
friend class distribution::uniform_cuda_hip::UniformReal;
template<typename T>
friend class distribution::uniform_cuda_hip::UniformUint;

#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
curandStateXORWOW_t state = curandStateXORWOW_t{};
#        else
hiprandStateXORWOW_t state = hiprandStateXORWOW_t{};
#        endif

public:
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
using result_type = decltype(curand(&state));
#        else
using result_type = decltype(hiprand(&state));
#        endif
ALPAKA_FN_HOST_ACC constexpr static result_type min()
{
return std::numeric_limits<result_type>::min();
}
ALPAKA_FN_HOST_ACC constexpr static result_type max()
{
return std::numeric_limits<result_type>::max();
}
__device__ result_type operator()()
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
return curand(&state);
#        else
return hiprand(&state);
#        endif
}
};
} 

namespace distribution::uniform_cuda_hip
{
template<>
class NormalReal<float>
{
public:
template<typename TEngine>
__device__ auto operator()(TEngine& engine) -> float
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
return curand_normal(&engine.state);
#        else
return hiprand_normal(&engine.state);
#        endif
}
};

template<>
class NormalReal<double>
{
public:
template<typename TEngine>
__device__ auto operator()(TEngine& engine) -> double
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
return curand_normal_double(&engine.state);
#        else
return hiprand_normal_double(&engine.state);
#        endif
}
};

template<>
class UniformReal<float>
{
public:
template<typename TEngine>
__device__ auto operator()(TEngine& engine) -> float
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
float const fUniformRand(curand_uniform(&engine.state));
#        else
float const fUniformRand(hiprand_uniform(&engine.state));
#        endif
return fUniformRand * static_cast<float>(fUniformRand != 1.0f);
}
};

template<>
class UniformReal<double>
{
public:
template<typename TEngine>
__device__ auto operator()(TEngine& engine) -> double
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
double const fUniformRand(curand_uniform_double(&engine.state));
#        else
double const fUniformRand(hiprand_uniform_double(&engine.state));
#        endif
return fUniformRand * static_cast<double>(fUniformRand != 1.0);
}
};

template<>
class UniformUint<unsigned int>
{
public:
template<typename TEngine>
__device__ auto operator()(TEngine& engine) -> unsigned int
{
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
return curand(&engine.state);
#        else
return hiprand(&engine.state);
#        endif
}
};
} 

namespace distribution::trait
{
template<typename TApi, typename T>
struct CreateNormalReal<RandUniformCudaHipRand<TApi>, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
__device__ static auto createNormalReal(RandUniformCudaHipRand<TApi> const& )
-> uniform_cuda_hip::NormalReal<T>
{
return {};
}
};

template<typename TApi, typename T>
struct CreateUniformReal<RandUniformCudaHipRand<TApi>, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
__device__ static auto createUniformReal(RandUniformCudaHipRand<TApi> const& )
-> uniform_cuda_hip::UniformReal<T>
{
return {};
}
};

template<typename TApi, typename T>
struct CreateUniformUint<RandUniformCudaHipRand<TApi>, T, std::enable_if_t<std::is_integral_v<T>>>
{
__device__ static auto createUniformUint(RandUniformCudaHipRand<TApi> const& )
-> uniform_cuda_hip::UniformUint<T>
{
return {};
}
};
} 

namespace engine::trait
{
template<typename TApi>
struct CreateDefault<RandUniformCudaHipRand<TApi>>
{
__device__ static auto createDefault(
RandUniformCudaHipRand<TApi> const& ,
std::uint32_t const& seed = 0,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0) -> uniform_cuda_hip::Xor
{
return {seed, subsequence, offset};
}
};
} 
#    endif
} 

#endif
