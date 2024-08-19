

#pragma once

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevUniformCudaHipRt.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewAccessOps.hpp>
#include <alpaka/vec/Vec.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
template<typename TDev, typename TElem, typename TDim, typename TIdx>
class ViewPlainPtr final : public internal::ViewAccessOps<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
static_assert(!std::is_const_v<TIdx>, "The idx type of the view can not be const!");

using Dev = alpaka::Dev<TDev>;

public:
template<typename TExtent>
ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, Dev dev, TExtent const& extent = TExtent())
: m_pMem(pMem)
, m_dev(std::move(dev))
, m_extentElements(getExtentVecEnd<TDim>(extent))
, m_pitchBytes(detail::calculatePitchesFromExtents<TElem>(m_extentElements))
{
}

template<typename TExtent, typename TPitch>
ALPAKA_FN_HOST ViewPlainPtr(TElem* pMem, Dev const dev, TExtent const& extent, TPitch const& pitchBytes)
: m_pMem(pMem)
, m_dev(dev)
, m_extentElements(getExtentVecEnd<TDim>(extent))
, m_pitchBytes(subVecEnd<TDim>(static_cast<Vec<TDim, TIdx>>(pitchBytes)))
{
}

ViewPlainPtr(ViewPlainPtr const&) = default;
ALPAKA_FN_HOST
ViewPlainPtr(ViewPlainPtr&& other) noexcept
: m_pMem(other.m_pMem)
, m_dev(other.m_dev)
, m_extentElements(other.m_extentElements)
, m_pitchBytes(other.m_pitchBytes)
{
}
ALPAKA_FN_HOST
auto operator=(ViewPlainPtr const&) -> ViewPlainPtr& = delete;
ALPAKA_FN_HOST
auto operator=(ViewPlainPtr&&) -> ViewPlainPtr& = delete;

public:
TElem* const m_pMem;
Dev const m_dev;
Vec<TDim, TIdx> const m_extentElements;
Vec<TDim, TIdx> const m_pitchBytes;
};

namespace trait
{
template<typename TDev, typename TElem, typename TDim, typename TIdx>
struct DevType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
using type = alpaka::Dev<TDev>;
};

template<typename TDev, typename TElem, typename TDim, typename TIdx>
struct GetDev<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
static auto getDev(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> alpaka::Dev<TDev>
{
return view.m_dev;
}
};

template<typename TDev, typename TElem, typename TDim, typename TIdx>
struct DimType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
using type = TDim;
};

template<typename TDev, typename TElem, typename TDim, typename TIdx>
struct ElemType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
using type = TElem;
};
} 
namespace trait
{
template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
struct GetExtent<
TIdxIntegralConst,
ViewPlainPtr<TDev, TElem, TDim, TIdx>,
std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
{
ALPAKA_FN_HOST
static auto getExtent(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& extent) -> TIdx
{
return extent.m_extentElements[TIdxIntegralConst::value];
}
};
} 

namespace trait
{
template<typename TDev, typename TElem, typename TDim, typename TIdx>
struct GetPtrNative<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
static auto getPtrNative(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> TElem const*
{
return view.m_pMem;
}
static auto getPtrNative(ViewPlainPtr<TDev, TElem, TDim, TIdx>& view) -> TElem*
{
return view.m_pMem;
}
};

template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
struct GetPitchBytes < TIdxIntegralConst,
ViewPlainPtr<TDev, TElem, TDim, TIdx>, std::enable_if_t<TIdxIntegralConst::value<TDim::value>>
{
ALPAKA_FN_HOST static auto getPitchBytes(ViewPlainPtr<TDev, TElem, TDim, TIdx> const& view) -> TIdx
{
return view.m_pitchBytes[TIdxIntegralConst::value];
}
};

template<>
struct CreateStaticDevMemView<DevCpu>
{
template<typename TElem, typename TExtent>
static auto createStaticDevMemView(TElem* pMem, DevCpu const& dev, TExtent const& extent)
{
return alpaka::ViewPlainPtr<DevCpu, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
pMem,
dev,
extent);
}
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template<typename TApi>
struct CreateStaticDevMemView<DevUniformCudaHipRt<TApi>>
{
template<typename TElem, typename TExtent>
static auto createStaticDevMemView(
TElem* pMem,
DevUniformCudaHipRt<TApi> const& dev,
TExtent const& extent)
{
TElem* pMemAcc(nullptr);
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(TApi::getSymbolAddress(reinterpret_cast<void**>(&pMemAcc), *pMem));

return alpaka::
ViewPlainPtr<DevUniformCudaHipRt<TApi>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
pMemAcc,
dev,
extent);
}
};
#endif

template<>
struct CreateViewPlainPtr<DevCpu>
{
template<typename TElem, typename TExtent, typename TPitch>
static auto createViewPlainPtr(DevCpu const& dev, TElem* pMem, TExtent const& extent, TPitch const& pitch)
{
return alpaka::ViewPlainPtr<DevCpu, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
pMem,
dev,
extent,
pitch);
}
};

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
template<typename TApi>
struct CreateViewPlainPtr<DevUniformCudaHipRt<TApi>>
{
template<typename TElem, typename TExtent, typename TPitch>
static auto createViewPlainPtr(
DevUniformCudaHipRt<TApi> const& dev,
TElem* pMem,
TExtent const& extent,
TPitch const& pitch)
{
return alpaka::
ViewPlainPtr<DevUniformCudaHipRt<TApi>, TElem, alpaka::Dim<TExtent>, alpaka::Idx<TExtent>>(
pMem,
dev,
extent,
pitch);
}
};
#endif

template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
struct GetOffset<TIdxIntegralConst, ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
ALPAKA_FN_HOST
static auto getOffset(ViewPlainPtr<TDev, TElem, TDim, TIdx> const&) -> TIdx
{
return 0u;
}
};

template<typename TDev, typename TElem, typename TDim, typename TIdx>
struct IdxType<ViewPlainPtr<TDev, TElem, TDim, TIdx>>
{
using type = TIdx;
};
} 
} 
