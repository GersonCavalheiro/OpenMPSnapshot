

#pragma once

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/view/ViewAccessOps.hpp>
#include <alpaka/mem/view/ViewPlainPtr.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <type_traits>
#include <utility>

namespace alpaka
{
template<typename TDev, typename TElem, typename TDim, typename TIdx>
class ViewSubView : public internal::ViewAccessOps<ViewSubView<TDev, TElem, TDim, TIdx>>
{
static_assert(!std::is_const_v<TIdx>, "The idx type of the view can not be const!");

using Dev = alpaka::Dev<TDev>;

public:
template<typename TView, typename TOffsets, typename TExtent>
ViewSubView(
TView const& view,
TExtent const& extentElements,
TOffsets const& relativeOffsetsElements = TOffsets())
: m_viewParentView(getPtrNative(view), getDev(view), getExtentVec(view), getPitchBytesVec(view))
, m_extentElements(getExtentVec(extentElements))
, m_offsetsElements(getOffsetVec(relativeOffsetsElements))
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

static_assert(
std::is_same_v<Dev, alpaka::Dev<TView>>,
"The dev type of TView and the Dev template parameter have to be identical!");

static_assert(
std::is_same_v<TIdx, Idx<TView>>,
"The idx type of TView and the TIdx template parameter have to be identical!");
static_assert(
std::is_same_v<TIdx, Idx<TExtent>>,
"The idx type of TExtent and the TIdx template parameter have to be identical!");
static_assert(
std::is_same_v<TIdx, Idx<TOffsets>>,
"The idx type of TOffsets and the TIdx template parameter have to be identical!");

static_assert(
std::is_same_v<TDim, Dim<TView>>,
"The dim type of TView and the TDim template parameter have to be identical!");
static_assert(
std::is_same_v<TDim, Dim<TExtent>>,
"The dim type of TExtent and the TDim template parameter have to be identical!");
static_assert(
std::is_same_v<TDim, Dim<TOffsets>>,
"The dim type of TOffsets and the TDim template parameter have to be identical!");

ALPAKA_ASSERT(((m_offsetsElements + m_extentElements) <= getExtentVec(view))
.foldrAll(std::logical_and<bool>(), true));
}
template<typename TView, typename TOffsets, typename TExtent>
ViewSubView(TView& view, TExtent const& extentElements, TOffsets const& relativeOffsetsElements = TOffsets())
: m_viewParentView(getPtrNative(view), getDev(view), getExtentVec(view), getPitchBytesVec(view))
, m_extentElements(getExtentVec(extentElements))
, m_offsetsElements(getOffsetVec(relativeOffsetsElements))
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;

static_assert(
std::is_same_v<Dev, alpaka::Dev<TView>>,
"The dev type of TView and the Dev template parameter have to be identical!");

static_assert(
std::is_same_v<TIdx, Idx<TView>>,
"The idx type of TView and the TIdx template parameter have to be identical!");
static_assert(
std::is_same_v<TIdx, Idx<TExtent>>,
"The idx type of TExtent and the TIdx template parameter have to be identical!");
static_assert(
std::is_same_v<TIdx, Idx<TOffsets>>,
"The idx type of TOffsets and the TIdx template parameter have to be identical!");

static_assert(
std::is_same_v<TDim, Dim<TView>>,
"The dim type of TView and the TDim template parameter have to be identical!");
static_assert(
std::is_same_v<TDim, Dim<TExtent>>,
"The dim type of TExtent and the TDim template parameter have to be identical!");
static_assert(
std::is_same_v<TDim, Dim<TOffsets>>,
"The dim type of TOffsets and the TDim template parameter have to be identical!");

ALPAKA_ASSERT(((m_offsetsElements + m_extentElements) <= getExtentVec(view))
.foldrAll(std::logical_and<bool>(), true));
}

template<typename TView>
explicit ViewSubView(TView const& view) : ViewSubView(view, view, Vec<TDim, TIdx>::all(0))
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;
}

template<typename TView>
explicit ViewSubView(TView& view) : ViewSubView(view, view, Vec<TDim, TIdx>::all(0))
{
ALPAKA_DEBUG_FULL_LOG_SCOPE;
}

public:
ViewPlainPtr<Dev, TElem, TDim, TIdx> m_viewParentView; 
Vec<TDim, TIdx> m_extentElements; 
Vec<TDim, TIdx> m_offsetsElements; 
};

namespace trait
{
template<typename TElem, typename TDim, typename TDev, typename TIdx>
struct DevType<ViewSubView<TDev, TElem, TDim, TIdx>>
{
using type = alpaka::Dev<TDev>;
};

template<typename TElem, typename TDim, typename TDev, typename TIdx>
struct GetDev<ViewSubView<TDev, TElem, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getDev(ViewSubView<TDev, TElem, TDim, TIdx> const& view) -> alpaka::Dev<TDev>
{
return alpaka::getDev(view.m_viewParentView);
}
};

template<typename TElem, typename TDim, typename TDev, typename TIdx>
struct DimType<ViewSubView<TDev, TElem, TDim, TIdx>>
{
using type = TDim;
};

template<typename TElem, typename TDim, typename TDev, typename TIdx>
struct ElemType<ViewSubView<TDev, TElem, TDim, TIdx>>
{
using type = TElem;
};

template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TDev, typename TIdx>
struct GetExtent<
TIdxIntegralConst,
ViewSubView<TDev, TElem, TDim, TIdx>,
std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
{
ALPAKA_FN_HOST static auto getExtent(ViewSubView<TDev, TElem, TDim, TIdx> const& extent) -> TIdx
{
return extent.m_extentElements[TIdxIntegralConst::value];
}
};

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored                                                                                    \
"-Wcast-align" 
#endif
template<typename TElem, typename TDim, typename TDev, typename TIdx>
struct GetPtrNative<ViewSubView<TDev, TElem, TDim, TIdx>>
{
private:
using IdxSequence = std::make_index_sequence<TDim::value>;

public:
ALPAKA_FN_HOST static auto getPtrNative(ViewSubView<TDev, TElem, TDim, TIdx> const& view) -> TElem const*
{
return reinterpret_cast<TElem const*>(
reinterpret_cast<std::uint8_t const*>(alpaka::getPtrNative(view.m_viewParentView))
+ pitchedOffsetBytes(view, IdxSequence()));
}
ALPAKA_FN_HOST static auto getPtrNative(ViewSubView<TDev, TElem, TDim, TIdx>& view) -> TElem*
{
return reinterpret_cast<TElem*>(
reinterpret_cast<std::uint8_t*>(alpaka::getPtrNative(view.m_viewParentView))
+ pitchedOffsetBytes(view, IdxSequence()));
}

private:
template<typename TView, std::size_t... TIndices>
ALPAKA_FN_HOST static auto pitchedOffsetBytes(TView const& view, std::index_sequence<TIndices...> const&)
-> TIdx
{
return meta::foldr(std::plus<TIdx>(), pitchedOffsetBytesDim<TIndices>(view)..., TIdx{0});
}
template<std::size_t Tidx, typename TView>
ALPAKA_FN_HOST static auto pitchedOffsetBytesDim(TView const& view) -> TIdx
{
return getOffset<Tidx>(view) * getPitchBytes<Tidx + 1u>(view);
}
};
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

template<typename TIdxIntegralConst, typename TDev, typename TElem, typename TDim, typename TIdx>
struct GetPitchBytes<TIdxIntegralConst, ViewSubView<TDev, TElem, TDim, TIdx>>
{
ALPAKA_FN_HOST static auto getPitchBytes(ViewSubView<TDev, TElem, TDim, TIdx> const& view) -> TIdx
{
return alpaka::getPitchBytes<TIdxIntegralConst::value>(view.m_viewParentView);
}
};

template<typename TIdxIntegralConst, typename TElem, typename TDim, typename TDev, typename TIdx>
struct GetOffset<
TIdxIntegralConst,
ViewSubView<TDev, TElem, TDim, TIdx>,
std::enable_if_t<(TDim::value > TIdxIntegralConst::value)>>
{
ALPAKA_FN_HOST static auto getOffset(ViewSubView<TDev, TElem, TDim, TIdx> const& offset) -> TIdx
{
return offset.m_offsetsElements[TIdxIntegralConst::value];
}
};

template<typename TElem, typename TDim, typename TDev, typename TIdx>
struct IdxType<ViewSubView<TDev, TElem, TDim, TIdx>>
{
using type = TIdx;
};

template<typename TDev, typename TSfinae>
struct CreateSubView
{
template<typename TView, typename TExtent, typename TOffsets>
static auto createSubView(
TView& view,
TExtent const& extentElements,
TOffsets const& relativeOffsetsElements)
{
using Dim = alpaka::Dim<TExtent>;
using Idx = alpaka::Idx<TExtent>;
using Elem = typename trait::ElemType<TView>::type;
return ViewSubView<TDev, Elem, Dim, Idx>(view, extentElements, relativeOffsetsElements);
}
};
} 
} 
