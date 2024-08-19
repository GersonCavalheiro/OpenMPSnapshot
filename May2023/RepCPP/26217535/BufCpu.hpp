

#pragma once

#include <alpaka/core/Vectorize.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/mem/view/ViewAccessOps.hpp>
#include <alpaka/vec/Vec.hpp>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    include <alpaka/core/ApiCudaRt.hpp>
#    include <alpaka/core/Cuda.hpp>
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#    include <alpaka/core/ApiHipRt.hpp>
#    include <alpaka/core/Hip.hpp>
#endif

#include <alpaka/mem/alloc/AllocCpuAligned.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace alpaka
{
namespace detail
{
template<typename TElem, typename TDim, typename TIdx>
class BufCpuImpl final
{
static_assert(
!std::is_const_v<TElem>,
"The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
"elements!");
static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

public:
template<typename TExtent>
ALPAKA_FN_HOST BufCpuImpl(
DevCpu dev,
TElem* pMem,
std::function<void(TElem*)> deleter,
TExtent const& extent) noexcept
: m_dev(std::move(dev))
, m_extentElements(getExtentVecEnd<TDim>(extent))
, m_pMem(pMem)
, m_deleter(std::move(deleter))
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

static_assert(
TDim::value == Dim<TExtent>::value,
"The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
"identical!");
static_assert(
std::is_same_v<TIdx, Idx<TExtent>>,
"The idx type of TExtent and the TIdx template parameter have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
std::cout << __func__ << " e: " << m_extentElements << " ptr: " << static_cast<void*>(m_pMem)
<< std::endl;
#endif
}
BufCpuImpl(BufCpuImpl&&) = delete;
auto operator=(BufCpuImpl&&) -> BufCpuImpl& = delete;
ALPAKA_FN_HOST ~BufCpuImpl()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

m_deleter(m_pMem);
}

public:
DevCpu const m_dev;
Vec<TDim, TIdx> const m_extentElements;
TElem* const m_pMem;
std::function<void(TElem*)> m_deleter;
};
} 

template<typename TElem, typename TDim, typename TIdx>
class BufCpu : public internal::ViewAccessOps<BufCpu<TElem, TDim, TIdx>>
{
public:
template<typename TExtent, typename Deleter>
ALPAKA_FN_HOST BufCpu(DevCpu const& dev, TElem* pMem, Deleter deleter, TExtent const& extent)
: m_spBufCpuImpl{
std::make_shared<detail::BufCpuImpl<TElem, TDim, TIdx>>(dev, pMem, std::move(deleter), extent)}
{
}

public:
std::shared_ptr<detail::BufCpuImpl<TElem, TDim, TIdx>> m_spBufCpuImpl;
};

namespace trait
{
template<typename TElem, typename TDim, typename TIdx>
struct DevType<BufCpu<TElem, TDim, TIdx>>
{
using type = DevCpu;
};
template<typename TElem, typename TDim, typename TIdx>
struct GetDev<BufCpu<TElem, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getDev(BufCpu<TElem, TDim, TIdx> const& buf) -> DevCpu
{
return buf.m_spBufCpuImpl->m_dev;
}
};

template<typename TElem, typename TDim, typename TIdx>
struct DimType<BufCpu<TElem, TDim, TIdx>>
{
using type = TDim;
};

template<typename TElem, typename TDim, typename TIdx>
struct ElemType<BufCpu<TElem, TDim, TIdx>>
{
using type = TElem;
};

template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
struct GetExtent<
TIdxIntegralConst,
BufCpu<TElem, TDim, TIdx>,
std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
{
ALPAKA_FN_HOST static auto getExtent(BufCpu<TElem, TDim, TIdx> const& extent) -> TIdx
{
return extent.m_spBufCpuImpl->m_extentElements[TIdxIntegralConst::value];
}
};

template<typename TElem, typename TDim, typename TIdx>
struct GetPtrNative<BufCpu<TElem, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx> const& buf) -> TElem const*
{
return buf.m_spBufCpuImpl->m_pMem;
}
ALPAKA_FN_HOST static auto getPtrNative(BufCpu<TElem, TDim, TIdx>& buf) -> TElem*
{
return buf.m_spBufCpuImpl->m_pMem;
}
};
template<typename TElem, typename TDim, typename TIdx>
struct GetPtrDev<BufCpu<TElem, TDim, TIdx>, DevCpu>
{
ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx> const& buf, DevCpu const& dev)
-> TElem const*
{
if(dev == getDev(buf))
{
return buf.m_spBufCpuImpl->m_pMem;
}
else
{
throw std::runtime_error("The buffer is not accessible from the given device!");
}
}
ALPAKA_FN_HOST static auto getPtrDev(BufCpu<TElem, TDim, TIdx>& buf, DevCpu const& dev) -> TElem*
{
if(dev == getDev(buf))
{
return buf.m_spBufCpuImpl->m_pMem;
}
else
{
throw std::runtime_error("The buffer is not accessible from the given device!");
}
}
};

template<typename TElem, typename TDim, typename TIdx>
struct BufAlloc<TElem, TDim, TIdx, DevCpu>
{
template<typename TExtent>
ALPAKA_FN_HOST static auto allocBuf(DevCpu const& dev, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#if defined(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT)
static_assert(
ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT > 0
&& ((ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT & (ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT - 1)) == 0),
"If defined, ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT must be a power of 2.");
constexpr std::size_t alignment = static_cast<std::size_t>(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT);
#else
constexpr std::size_t alignment = core::vectorization::defaultAlignment;
#endif
using Allocator = AllocCpuAligned<std::integral_constant<std::size_t, alignment>>;
static_assert(std::is_empty_v<Allocator>, "AllocCpuAligned is expected to be stateless");
auto* memPtr = alpaka::malloc<TElem>(Allocator{}, static_cast<std::size_t>(getExtentProduct(extent)));
auto deleter = [](TElem* ptr) { alpaka::free(Allocator{}, ptr); };

return BufCpu<TElem, TDim, TIdx>(dev, memPtr, std::move(deleter), extent);
}
};
template<typename TElem, typename TDim, typename TIdx>
struct AsyncBufAlloc<TElem, TDim, TIdx, DevCpu>
{
template<typename TQueue, typename TExtent>
ALPAKA_FN_HOST static auto allocAsyncBuf(TQueue queue, TExtent const& extent) -> BufCpu<TElem, TDim, TIdx>
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

static_assert(
std::is_same_v<Dev<TQueue>, DevCpu>,
"The BufCpu buffer can only be used with a queue on a DevCpu device!");
DevCpu const& dev = getDev(queue);

#if defined(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT)
static_assert(
ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT > 0
&& ((ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT & (ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT - 1)) == 0),
"If defined, ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT must be a power of 2.");
constexpr std::size_t alignment = static_cast<std::size_t>(ALPAKA_DEFAULT_HOST_MEMORY_ALIGNMENT);
#else
constexpr std::size_t alignment = core::vectorization::defaultAlignment;
#endif
using Allocator = AllocCpuAligned<std::integral_constant<std::size_t, alignment>>;
static_assert(std::is_empty_v<Allocator>, "AllocCpuAligned is expected to be stateless");
auto* memPtr = alpaka::malloc<TElem>(Allocator{}, static_cast<std::size_t>(getExtentProduct(extent)));
auto deleter = [queue = std::move(queue)](TElem* ptr) mutable
{
alpaka::enqueue(
queue,
[ptr]()
{
alpaka::free(Allocator{}, ptr);
});
};

return BufCpu<TElem, TDim, TIdx>(dev, memPtr, std::move(deleter), extent);
}
};

template<typename TDim>
struct HasAsyncBufSupport<TDim, DevCpu> : public std::true_type
{
};

template<typename TElem, typename TDim, typename TIdx>
struct BufAllocMapped<PltfCpu, TElem, TDim, TIdx>
{
template<typename TExtent>
ALPAKA_FN_HOST static auto allocMappedBuf(DevCpu const& host, TExtent const& extent)
-> BufCpu<TElem, TDim, TIdx>
{
return allocBuf<TElem, TIdx>(host, extent);
}
};

template<>
struct HasMappedBufSupport<PltfCpu> : public std::true_type
{
};

template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TIdx>
struct GetOffset<TIdxIntegralConst, BufCpu<TElem, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getOffset(BufCpu<TElem, TDim, TIdx> const&) -> TIdx
{
return 0u;
}
};

template<typename TElem, typename TDim, typename TIdx>
struct IdxType<BufCpu<TElem, TDim, TIdx>>
{
using type = TIdx;
};
} 
} 

#include <alpaka/mem/buf/cpu/Copy.hpp>
#include <alpaka/mem/buf/cpu/Set.hpp>