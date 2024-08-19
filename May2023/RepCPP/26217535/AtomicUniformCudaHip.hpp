

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/atomic/Op.hpp>
#    include <alpaka/core/BoostPredef.hpp>
#    include <alpaka/core/Positioning.hpp>
#    include <alpaka/core/Utility.hpp>

#    include <type_traits>

namespace alpaka
{
class AtomicUniformCudaHipBuiltIn
{
};
} 

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

#        define CLANG_CUDA_PTX_WORKAROUND                                                                             \
(BOOST_COMP_CLANG && BOOST_LANG_CUDA && BOOST_ARCH_PTX < BOOST_VERSION_NUMBER(6, 0, 0))

inline namespace alpakaGlobal
{
template<typename TOp, typename T, typename THierarchy, typename TSfinae = void>
struct AlpakaBuiltInAtomic : std::false_type
{
};

template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicCas,
T,
THierarchy,
typename std::void_t<
decltype(atomicCAS(alpaka::core::declval<T*>(), alpaka::core::declval<T>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T compare, T value)
{
return atomicCAS(add, compare, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicCas,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicCAS_block(
alpaka::core::declval<T*>(),
alpaka::core::declval<T>(),
alpaka::core::declval<T>()))>> : std::true_type
{
__device__ static T atomic(T* add, T compare, T value)
{
return atomicCAS_block(add, compare, value);
}
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicAdd,
T,
THierarchy,
typename std::void_t<decltype(atomicAdd(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicAdd(add, value);
}
};


#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicAdd,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicAdd_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicAdd_block(add, value);
}
};
#        endif

#        if CLANG_CUDA_PTX_WORKAROUND
template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicAdd, double, THierarchy> : std::false_type
{
};
#        endif

#        if(BOOST_LANG_HIP)
template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicAdd, float, alpaka::hierarchy::Threads> : std::false_type
{
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicSub,
T,
THierarchy,
typename std::void_t<decltype(atomicSub(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicSub(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicSub,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicSub_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicSub_block(add, value);
}
};
#        endif

template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicMin,
T,
THierarchy,
typename std::void_t<decltype(atomicMin(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicMin(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicMin,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicMin_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicMin_block(add, value);
}
};
#        endif

#        if(BOOST_LANG_HIP)
template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicMin, float, THierarchy> : std::false_type
{
};

template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicMin, float, alpaka::hierarchy::Threads> : std::false_type
{
};

template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicMin, double, THierarchy> : std::false_type
{
};

template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicMin, double, alpaka::hierarchy::Threads> : std::false_type
{
};

#            if !__has_builtin(__hip_atomic_compare_exchange_strong)
template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicMin, unsigned long long, THierarchy> : std::false_type
{
};

template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicMin, unsigned long long, alpaka::hierarchy::Threads> : std::false_type
{
};
#            endif
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicMax,
T,
THierarchy,
typename std::void_t<decltype(atomicMax(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicMax(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicMax,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicMax_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicMax_block(add, value);
}
};
#        endif

#        if(BOOST_LANG_HIP)
template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicMax, float, THierarchy> : std::false_type
{
};

template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicMax, float, alpaka::hierarchy::Threads> : std::false_type
{
};

template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicMax, double, THierarchy> : std::false_type
{
};

template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicMax, double, alpaka::hierarchy::Threads> : std::false_type
{
};

#            if !__has_builtin(__hip_atomic_compare_exchange_strong)
template<typename THierarchy>
struct AlpakaBuiltInAtomic<alpaka::AtomicMax, unsigned long long, THierarchy> : std::false_type
{
};

template<>
struct AlpakaBuiltInAtomic<alpaka::AtomicMax, unsigned long long, alpaka::hierarchy::Threads> : std::false_type
{
};
#            endif
#        endif



template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicExch,
T,
THierarchy,
typename std::void_t<decltype(atomicExch(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicExch(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicExch,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicExch_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicExch_block(add, value);
}
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicInc,
T,
THierarchy,
typename std::void_t<decltype(atomicInc(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicInc(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicInc,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicInc_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicInc_block(add, value);
}
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicDec,
T,
THierarchy,
typename std::void_t<decltype(atomicDec(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicDec(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicDec,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicDec_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicDec_block(add, value);
}
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicAnd,
T,
THierarchy,
typename std::void_t<decltype(atomicAnd(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicAnd(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicAnd,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicAnd_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicAnd_block(add, value);
}
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicOr,
T,
THierarchy,
typename std::void_t<decltype(atomicOr(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicOr(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicOr,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicOr_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicOr_block(add, value);
}
};
#        endif


template<typename T, typename THierarchy>
struct AlpakaBuiltInAtomic<
alpaka::AtomicXor,
T,
THierarchy,
typename std::void_t<decltype(atomicXor(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicXor(add, value);
}
};

#        if !CLANG_CUDA_PTX_WORKAROUND
template<typename T>
struct AlpakaBuiltInAtomic<
alpaka::AtomicXor,
T,
alpaka::hierarchy::Threads,
typename std::void_t<decltype(atomicXor_block(alpaka::core::declval<T*>(), alpaka::core::declval<T>()))>>
: std::true_type
{
__device__ static T atomic(T* add, T value)
{
return atomicXor_block(add, value);
}
};
#        endif

} 

#        undef CLANG_CUDA_PTX_WORKAROUND
#    endif

#endif
