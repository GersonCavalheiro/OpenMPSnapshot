

#pragma once

#include <alpaka/acc/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Utility.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <set>
#include <type_traits>

namespace alpaka
{
enum class GridBlockExtentSubDivRestrictions
{
EqualExtent, 
CloseToEqualExtent, 
Unrestricted, 
};

namespace detail
{
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
ALPAKA_FN_HOST auto nextDivisorLowerOrEqual(T const& dividend, T const& maxDivisor) -> T
{
core::assertValueUnsigned(dividend);
core::assertValueUnsigned(maxDivisor);
ALPAKA_ASSERT(dividend >= maxDivisor);

T divisor = maxDivisor;
while(dividend % divisor != 0)
--divisor;
return divisor;
}
template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
ALPAKA_FN_HOST auto allDivisorsLessOrEqual(T const& val, T const& maxDivisor) -> std::set<T>
{
std::set<T> divisorSet;

core::assertValueUnsigned(val);
core::assertValueUnsigned(maxDivisor);
ALPAKA_ASSERT(maxDivisor <= val);

for(T i(1); i <= std::min(val, maxDivisor); ++i)
{
if(val % i == 0)
{
divisorSet.insert(static_cast<T>(val / i));
}
}

return divisorSet;
}
} 

template<typename TDim, typename TIdx>
ALPAKA_FN_HOST auto isValidAccDevProps(AccDevProps<TDim, TIdx> const& accDevProps) -> bool
{
if((accDevProps.m_gridBlockCountMax < 1) || (accDevProps.m_blockThreadCountMax < 1)
|| (accDevProps.m_threadElemCountMax < 1))
{
return false;
}

auto const gridBlockExtentMax = subVecEnd<TDim>(accDevProps.m_gridBlockExtentMax);
auto const blockThreadExtentMax = subVecEnd<TDim>(accDevProps.m_blockThreadExtentMax);
auto const threadElemExtentMax = subVecEnd<TDim>(accDevProps.m_threadElemExtentMax);

for(typename TDim::value_type i(0); i < TDim::value; ++i)
{
if((gridBlockExtentMax[i] < 1) || (blockThreadExtentMax[i] < 1) || (threadElemExtentMax[i] < 1))
{
return false;
}
}

return true;
}

template<typename TDim, typename TIdx>
ALPAKA_FN_HOST auto subDivideGridElems(
Vec<TDim, TIdx> const& gridElemExtent,
Vec<TDim, TIdx> const& threadElemExtent,
AccDevProps<TDim, TIdx> const& accDevProps,
bool blockThreadMustDivideGridThreadExtent = true,
GridBlockExtentSubDivRestrictions gridBlockExtentSubDivRestrictions
= GridBlockExtentSubDivRestrictions::Unrestricted) -> WorkDivMembers<TDim, TIdx>
{
using Vec = Vec<TDim, TIdx>;
using DimLoopInd = typename TDim::value_type;

for(DimLoopInd i(0); i < TDim::value; ++i)
{
ALPAKA_ASSERT(gridElemExtent[i] >= 1);
ALPAKA_ASSERT(threadElemExtent[i] >= 1);
ALPAKA_ASSERT(threadElemExtent[i] <= accDevProps.m_threadElemExtentMax[i]);
}
ALPAKA_ASSERT(threadElemExtent.prod() <= accDevProps.m_threadElemCountMax);
ALPAKA_ASSERT(isValidAccDevProps(accDevProps));

auto const clippedThreadElemExtent = elementwise_min(threadElemExtent, gridElemExtent);
auto const gridThreadExtent = [&]
{
Vec r;
for(DimLoopInd i(0u); i < TDim::value; ++i)
r[i] = core::divCeil(gridElemExtent[i], clippedThreadElemExtent[i]);
return r;
}();


auto blockThreadExtent = elementwise_min(accDevProps.m_blockThreadExtentMax, gridThreadExtent);

if(gridBlockExtentSubDivRestrictions == GridBlockExtentSubDivRestrictions::EqualExtent)
blockThreadExtent = Vec::all(blockThreadExtent.min());

auto const& blockThreadCountMax = accDevProps.m_blockThreadCountMax;
if(blockThreadExtent.prod() > blockThreadCountMax)
{
switch(gridBlockExtentSubDivRestrictions)
{
case GridBlockExtentSubDivRestrictions::EqualExtent:
blockThreadExtent = Vec::all(core::nthRootFloor(blockThreadCountMax, TIdx{TDim::value}));
break;
case GridBlockExtentSubDivRestrictions::CloseToEqualExtent:
while(blockThreadExtent.prod() > blockThreadCountMax)
blockThreadExtent[blockThreadExtent.maxElem()] /= TIdx{2};
break;
case GridBlockExtentSubDivRestrictions::Unrestricted:
while(blockThreadExtent.prod() > blockThreadCountMax)
{
auto const it = std::min_element(
blockThreadExtent.begin(),
blockThreadExtent.end() - 1, 
[](TIdx const& a, TIdx const& b)
{
if(a == TIdx{1})
return false;
if(b == TIdx{1})
return true;
return a < b;
});
*it /= TIdx{2};
}
break;
}
}

if(blockThreadMustDivideGridThreadExtent)
{
switch(gridBlockExtentSubDivRestrictions)
{
case GridBlockExtentSubDivRestrictions::EqualExtent:
{
std::array<std::set<TIdx>, TDim::value> gridThreadExtentDivisors;
for(DimLoopInd i(0u); i < TDim::value; ++i)
{
gridThreadExtentDivisors[i]
= detail::allDivisorsLessOrEqual(gridThreadExtent[i], blockThreadExtent[i]);
}
std::set<TIdx> intersects[2u];
for(DimLoopInd i(1u); i < TDim::value; ++i)
{
intersects[(i - 1u) % 2u] = gridThreadExtentDivisors[0];
intersects[(i) % 2u].clear();
set_intersection(
std::begin(intersects[(i - 1u) % 2u]),
std::end(intersects[(i - 1u) % 2u]),
std::begin(gridThreadExtentDivisors[i]),
std::end(gridThreadExtentDivisors[i]),
std::inserter(intersects[i % 2], std::begin(intersects[i % 2u])));
}
TIdx const maxCommonDivisor = *(--std::end(intersects[(TDim::value - 1) % 2u]));
blockThreadExtent = Vec::all(maxCommonDivisor);
break;
}
case GridBlockExtentSubDivRestrictions::CloseToEqualExtent:
[[fallthrough]];
case GridBlockExtentSubDivRestrictions::Unrestricted:
for(DimLoopInd i(0u); i < TDim::value; ++i)
blockThreadExtent[i] = detail::nextDivisorLowerOrEqual(gridThreadExtent[i], blockThreadExtent[i]);
break;
}
}

auto const gridBlockExtent = [&]
{
Vec r;
for(DimLoopInd i = 0; i < TDim::value; ++i)
r[i] = core::divCeil(gridThreadExtent[i], blockThreadExtent[i]);
return r;
}();

return WorkDivMembers<TDim, TIdx>(gridBlockExtent, blockThreadExtent, clippedThreadElemExtent);
}

template<
typename TAcc,
typename TDev,
typename TGridElemExtent = Vec<Dim<TAcc>, Idx<TAcc>>,
typename TThreadElemExtent = Vec<Dim<TAcc>, Idx<TAcc>>>
ALPAKA_FN_HOST auto getValidWorkDiv(
[[maybe_unused]] TDev const& dev,
[[maybe_unused]] TGridElemExtent const& gridElemExtent = Vec<Dim<TAcc>, Idx<TAcc>>::ones(),
[[maybe_unused]] TThreadElemExtent const& threadElemExtents = Vec<Dim<TAcc>, Idx<TAcc>>::ones(),
[[maybe_unused]] bool blockThreadMustDivideGridThreadExtent = true,
[[maybe_unused]] GridBlockExtentSubDivRestrictions gridBlockExtentSubDivRestrictions
= GridBlockExtentSubDivRestrictions::Unrestricted)
-> WorkDivMembers<Dim<TGridElemExtent>, Idx<TGridElemExtent>>
{
static_assert(
Dim<TGridElemExtent>::value == Dim<TAcc>::value,
"The dimension of TAcc and the dimension of TGridElemExtent have to be identical!");
static_assert(
Dim<TThreadElemExtent>::value == Dim<TAcc>::value,
"The dimension of TAcc and the dimension of TThreadElemExtent have to be identical!");
static_assert(
std::is_same_v<Idx<TGridElemExtent>, Idx<TAcc>>,
"The idx type of TAcc and the idx type of TGridElemExtent have to be identical!");
static_assert(
std::is_same_v<Idx<TThreadElemExtent>, Idx<TAcc>>,
"The idx type of TAcc and the idx type of TThreadElemExtent have to be identical!");

if constexpr(Dim<TGridElemExtent>::value == 0)
{
auto const zero = Vec<DimInt<0>, Idx<TAcc>>{};
ALPAKA_ASSERT(gridElemExtent == zero);
ALPAKA_ASSERT(threadElemExtents == zero);
return WorkDivMembers<DimInt<0>, Idx<TAcc>>{zero, zero, zero};
}
else
return subDivideGridElems(
getExtentVec(gridElemExtent),
getExtentVec(threadElemExtents),
getAccDevProps<TAcc>(dev),
blockThreadMustDivideGridThreadExtent,
gridBlockExtentSubDivRestrictions);
using V [[maybe_unused]] = Vec<Dim<TGridElemExtent>, Idx<TGridElemExtent>>;
ALPAKA_UNREACHABLE(WorkDivMembers<Dim<TGridElemExtent>, Idx<TGridElemExtent>>{V{}, V{}, V{}});
}

template<typename TDim, typename TIdx, typename TWorkDiv>
ALPAKA_FN_HOST auto isValidWorkDiv(AccDevProps<TDim, TIdx> const& accDevProps, TWorkDiv const& workDiv) -> bool
{
auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(workDiv);
auto const blockThreadExtent = getWorkDiv<Block, Threads>(workDiv);
auto const threadElemExtent = getWorkDiv<Block, Threads>(workDiv);

if(accDevProps.m_gridBlockCountMax < gridBlockExtent.prod())
{
return false;
}
if(accDevProps.m_blockThreadCountMax < blockThreadExtent.prod())
{
return false;
}
if(accDevProps.m_threadElemCountMax < threadElemExtent.prod())
{
return false;
}

if constexpr(Dim<TWorkDiv>::value > 0)
{
auto const gridBlockExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_gridBlockExtentMax);
auto const blockThreadExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_blockThreadExtentMax);
auto const threadElemExtentMax = subVecEnd<Dim<TWorkDiv>>(accDevProps.m_threadElemExtentMax);

for(typename Dim<TWorkDiv>::value_type i(0); i < Dim<TWorkDiv>::value; ++i)
{
if((gridBlockExtent[i] < 1) || (blockThreadExtent[i] < 1) || (threadElemExtent[i] < 1)
|| (gridBlockExtentMax[i] < gridBlockExtent[i]) || (blockThreadExtentMax[i] < blockThreadExtent[i])
|| (threadElemExtentMax[i] < threadElemExtent[i]))
{
return false;
}
}
}

return true;
}
template<typename TAcc, typename TDev, typename TWorkDiv>
ALPAKA_FN_HOST auto isValidWorkDiv(TDev const& dev, TWorkDiv const& workDiv) -> bool
{
return isValidWorkDiv(getAccDevProps<TAcc>(dev), workDiv);
}
} 
