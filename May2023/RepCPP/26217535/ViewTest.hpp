

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/mem/view/Iterator.hpp>

#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <type_traits>

namespace alpaka::test
{
template<typename TElem, typename TDim, typename TIdx, typename TDev, typename TView>
ALPAKA_FN_HOST auto testViewImmutable(
TView const& view,
TDev const& dev,
Vec<TDim, TIdx> const& extent,
Vec<TDim, TIdx> const& offset) -> void
{
{
static_assert(
std::is_same_v<Dev<TView>, TDev>,
"The device type of the view has to be equal to the specified one.");
}

{
REQUIRE(dev == getDev(view));
}

{
static_assert(
Dim<TView>::value == TDim::value,
"The dimensionality of the view has to be equal to the specified one.");
}

{
static_assert(
std::is_same_v<Elem<TView>, TElem>,
"The element type of the view has to be equal to the specified one.");
}

{
REQUIRE(extent == getExtentVec(view));
}

{
auto pitchMinimum = Vec<DimInt<TDim::value + 1u>, TIdx>::ones();
pitchMinimum[TDim::value] = sizeof(TElem);
for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
{
pitchMinimum[i - 1] = extent[i - 1] * pitchMinimum[i];
}

auto const pitchView = getPitchBytesVec(view);

for(TIdx i = TDim::value; i > static_cast<TIdx>(0u); --i)
{
REQUIRE(pitchView[i - 1] >= pitchMinimum[i - 1]);
}
}

{
using NativePtr = decltype(getPtrNative(view));
static_assert(std::is_pointer_v<NativePtr>, "The value returned by getPtrNative has to be a pointer.");
static_assert(
std::is_const_v<std::remove_pointer_t<NativePtr>>,
"The value returned by getPtrNative has to be const when the view is const.");

if(getExtentProduct(view) != static_cast<TIdx>(0u))
{
TElem const* const invalidPtr(nullptr);
REQUIRE(invalidPtr != getPtrNative(view));
}
else
{
getPtrNative(view);
}
}

{
REQUIRE(offset == getOffsetVec(view));
}

{
static_assert(
std::is_same_v<Idx<TView>, TIdx>,
"The idx type of the view has to be equal to the specified one.");
}
}

struct VerifyBytesSetKernel
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TIter>
ALPAKA_FN_ACC void operator()(
TAcc const& acc [[maybe_unused]], 
bool* success,
TIter const& begin,
TIter const& end,
std::uint8_t const& byte) const
{
constexpr auto elemSizeInByte = sizeof(decltype(*begin));
for(auto it = begin; it != end; ++it)
{
auto const& elem = *it;
auto const pBytes = reinterpret_cast<std::uint8_t const*>(&elem);
for(std::size_t i = 0u; i < elemSizeInByte; ++i)
{
ALPAKA_CHECK(*success, pBytes[i] == byte);
}
}
}
};
template<typename TAcc, typename TView>
ALPAKA_FN_HOST auto verifyBytesSet(TView const& view, std::uint8_t const& byte) -> void
{
using Dim = Dim<TView>;
using Idx = Idx<TView>;

KernelExecutionFixture<TAcc> fixture(Vec<Dim, Idx>::ones());

VerifyBytesSetKernel verifyBytesSet;

REQUIRE(fixture(verifyBytesSet, test::begin(view), test::end(view), byte));
}

#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal" 
#endif
struct VerifyViewsEqualKernel
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename TAcc, typename TIterA, typename TIterB>
ALPAKA_FN_ACC void operator()(
TAcc const& acc [[maybe_unused]], 
bool* success,
TIterA beginA,
TIterA const& endA,
TIterB beginB) const
{
for(; beginA != endA; ++beginA, ++beginB)
{
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wfloat-equal" 
#endif
ALPAKA_CHECK(*success, *beginA == *beginB);
#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
}
}
};
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif

template<typename TAcc, typename TViewB, typename TViewA>
ALPAKA_FN_HOST auto verifyViewsEqual(TViewA const& viewA, TViewB const& viewB) -> void
{
using DimA = Dim<TViewA>;
using DimB = Dim<TViewB>;
static_assert(DimA::value == DimB::value, "viewA and viewB are required to have identical Dim");
using IdxA = Idx<TViewA>;
using IdxB = Idx<TViewB>;
static_assert(std::is_same_v<IdxA, IdxB>, "viewA and viewB are required to have identical Idx");

test::KernelExecutionFixture<TAcc> fixture(Vec<DimA, IdxA>::ones());

VerifyViewsEqualKernel verifyViewsEqualKernel;

REQUIRE(fixture(verifyViewsEqualKernel, test::begin(viewA), test::end(viewA), test::begin(viewB)));
}

template<typename TView, typename TQueue>
ALPAKA_FN_HOST auto iotaFillView(TQueue& queue, TView& view) -> void
{
using DevHost = DevCpu;
using PltfHost = Pltf<DevHost>;

using Elem = Elem<TView>;

DevHost const devHost = getDevByIdx<PltfHost>(0);

auto const extent = getExtentVec(view);

std::vector<Elem> v(static_cast<std::size_t>(extent.prod()), static_cast<Elem>(0));
std::iota(std::begin(v), std::end(v), static_cast<Elem>(0));
auto plainBuf = createView(devHost, v, extent);

memcpy(queue, view, plainBuf);

wait(queue);
}

template<typename TAcc, typename TView, typename TQueue>
ALPAKA_FN_HOST auto testViewMutable(TQueue& queue, TView& view) -> void
{
{
using NativePtr = decltype(getPtrNative(view));
static_assert(std::is_pointer_v<NativePtr>, "The value returned by getPtrNative has to be a pointer.");
static_assert(
!std::is_const_v<std::remove_pointer_t<NativePtr>>,
"The value returned by getPtrNative has to be non-const when the view is non-const.");
}

{
auto const byte(static_cast<uint8_t>(42u));
memset(queue, view, byte);
wait(queue);
verifyBytesSet<TAcc>(view, byte);
}

{
using Elem = Elem<TView>;
using Idx = Idx<TView>;

auto const devAcc = getDev(view);
auto const extent = getExtentVec(view);

{
auto srcBufAcc = allocBuf<Elem, Idx>(devAcc, extent);
iotaFillView(queue, srcBufAcc);
memcpy(queue, view, srcBufAcc);
wait(queue);
verifyViewsEqual<TAcc>(view, srcBufAcc);
}

{
auto dstBufAcc = allocBuf<Elem, Idx>(devAcc, extent);
memcpy(queue, dstBufAcc, view);
wait(queue);
verifyViewsEqual<TAcc>(dstBufAcc, view);
}
}
}
} 
