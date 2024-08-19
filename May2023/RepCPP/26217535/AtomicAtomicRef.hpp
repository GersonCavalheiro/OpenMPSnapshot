

#pragma once

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/core/BoostPredef.hpp>

#if defined(ALPAKA_HAS_STD_ATOMIC_REF)
#    include <atomic>
#else
#    include <boost/atomic.hpp>
#endif

#include <array>
#include <type_traits>

namespace alpaka
{
namespace detail
{
#if defined(ALPAKA_HAS_STD_ATOMIC_REF)
template<typename T>
using atomic_ref = std::atomic_ref<T>;
#else
template<typename T>
using atomic_ref = boost::atomic_ref<T>;
#endif
} 

class AtomicAtomicRef
{
};

template<typename T>
void isSupportedByAtomicAtomicRef()
{
static_assert(
std::is_trivially_copyable_v<T> && detail::atomic_ref<T>::required_alignment <= alignof(T),
"Type not supported by AtomicAtomicRef, please recompile defining "
"ALPAKA_DISABLE_ATOMIC_ATOMICREF.");
}

namespace trait
{
template<typename T, typename THierarchy>
struct AtomicOp<AtomicAdd, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
return ref.fetch_add(value);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicSub, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
return ref.fetch_sub(value);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicMin, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
T old = ref;
T result = old;
result = std::min(result, value);
while(!ref.compare_exchange_weak(old, result))
{
result = old;
result = std::min(result, value);
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicMax, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
T old = ref;
T result = old;
result = std::max(result, value);
while(!ref.compare_exchange_weak(old, result))
{
result = old;
result = std::max(result, value);
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicExch, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
T old = ref;
T result = value;
while(!ref.compare_exchange_weak(old, result))
{
result = value;
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicInc, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
T old = ref;
T result = ((old >= value) ? 0 : static_cast<T>(old + 1));
while(!ref.compare_exchange_weak(old, result))
{
result = ((old >= value) ? 0 : static_cast<T>(old + 1));
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicDec, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
T old = ref;
T result = ((old >= value) ? 0 : static_cast<T>(old - 1));
while(!ref.compare_exchange_weak(old, result))
{
result = ((old >= value) ? 0 : static_cast<T>(old - 1));
}
return old;
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicAnd, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
return ref.fetch_and(value);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicOr, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
return ref.fetch_or(value);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicXor, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicAtomicRef const&, T* const addr, T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
return ref.fetch_xor(value);
}
};

template<typename T, typename THierarchy>
struct AtomicOp<AtomicCas, AtomicAtomicRef, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(
AtomicAtomicRef const&,
T* const addr,
T const& compare,
T const& value) -> T
{
isSupportedByAtomicAtomicRef<T>();
detail::atomic_ref<T> ref(*addr);
T old = ref;
T result;
do
{
result = ((old == compare) ? value : old);
} while(!ref.compare_exchange_weak(old, result));
return old;
}
};
} 
} 
