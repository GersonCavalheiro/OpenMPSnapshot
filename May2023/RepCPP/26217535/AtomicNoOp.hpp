

#pragma once

#include <alpaka/atomic/Traits.hpp>

namespace alpaka
{
class AtomicNoOp
{
};

namespace trait
{
template<typename TOp, typename T, typename THierarchy>
struct AtomicOp<TOp, AtomicNoOp, T, THierarchy>
{
ALPAKA_FN_HOST static auto atomicOp(AtomicNoOp const& , T* const addr, T const& value) -> T
{
return TOp()(addr, value);
}

ALPAKA_FN_HOST static auto atomicOp(
AtomicNoOp const& ,
T* const addr,
T const& compare,
T const& value) -> T
{
return TOp()(addr, compare, value);
}
};
} 
} 
