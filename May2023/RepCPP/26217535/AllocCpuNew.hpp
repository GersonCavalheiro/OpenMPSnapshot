

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/mem/alloc/Traits.hpp>

namespace alpaka
{
class AllocCpuNew : public concepts::Implements<ConceptMemAlloc, AllocCpuNew>
{
};

namespace trait
{
template<typename T>
struct Malloc<T, AllocCpuNew>
{
ALPAKA_FN_HOST static auto malloc(AllocCpuNew const& , std::size_t const& sizeElems) -> T*
{
return new T[sizeElems];
}
};

template<typename T>
struct Free<T, AllocCpuNew>
{
ALPAKA_FN_HOST static auto free(AllocCpuNew const& , T const* const ptr) -> void
{
return delete[] ptr;
}
};
} 
} 
