

#pragma once

#include <alpaka/core/AlignedAlloc.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/dev/cpu/SysInfo.hpp>
#include <alpaka/mem/alloc/Traits.hpp>

#include <algorithm>

namespace alpaka
{
template<typename TAlignment>
class AllocCpuAligned : public concepts::Implements<ConceptMemAlloc, AllocCpuAligned<TAlignment>>
{
};

namespace trait
{
template<typename T, typename TAlignment>
struct Malloc<T, AllocCpuAligned<TAlignment>>
{
ALPAKA_FN_HOST static auto malloc(
AllocCpuAligned<TAlignment> const& ,
std::size_t const& sizeElems) -> T*
{
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
size_t minAlignement = std::max<size_t>(TAlignment::value, cpu::detail::getPageSize());
#else
constexpr size_t minAlignement = TAlignment::value;
#endif
return reinterpret_cast<T*>(core::alignedAlloc(minAlignement, sizeElems * sizeof(T)));
}
};

template<typename T, typename TAlignment>
struct Free<T, AllocCpuAligned<TAlignment>>
{
ALPAKA_FN_HOST static auto free(AllocCpuAligned<TAlignment> const& , T const* const ptr) -> void
{
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
size_t minAlignement = std::max<size_t>(TAlignment::value, cpu::detail::getPageSize());
#else
constexpr size_t minAlignement = TAlignment::value;
#endif
core::alignedFree(minAlignement, const_cast<void*>(reinterpret_cast<void const*>(ptr)));
}
};
} 
} 
