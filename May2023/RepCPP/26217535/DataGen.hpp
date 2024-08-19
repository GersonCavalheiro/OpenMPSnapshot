

#pragma once

#include "Defines.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <random>

namespace alpaka
{
namespace test
{
namespace unit
{
namespace math
{
template<typename TData>
struct RngWrapper
{
auto getMax()
{
return std::numeric_limits<TData>::max();
}

auto getLowest()
{
return std::numeric_limits<TData>::lowest();
}

auto getDistribution()
{
return std::uniform_real_distribution<TData>{0, 1000};
}

template<typename TDistribution, typename TEngine>
auto getNumber(TDistribution& distribution, TEngine& engine)
{
return distribution(engine);
}
};

template<typename TData>
struct RngWrapper<Complex<TData>>
{
auto getMax()
{
return Complex<TData>{TData{10}, TData{10}};
}

auto getLowest()
{
return -getMax();
}

auto getDistribution()
{
return std::uniform_real_distribution<TData>{0, 5};
}

template<typename TDistribution, typename TEngine>
auto getNumber(TDistribution& distribution, TEngine& engine)
{
return Complex<TData>{distribution(engine), distribution(engine)};
}
};


template<typename TData, typename TArgs, typename TFunctor>
auto fillWithRndArgs(TArgs& args, TFunctor functor, unsigned int const& seed) -> void
{

static_assert(
TArgs::value_type::arity == TFunctor::arity,
"Buffer properties must match TFunctor::arity");
static_assert(TArgs::capacity > 6, "Set of args must provide > 6 entries.");
auto rngWrapper = RngWrapper<TData>{};
auto const max = rngWrapper.getMax();
auto const low = rngWrapper.getLowest();
std::default_random_engine eng{static_cast<std::default_random_engine::result_type>(seed)};

auto dist = rngWrapper.getDistribution();
decltype(dist) distOne(-1, 1);
for(size_t k = 0; k < TFunctor::arity_nr; ++k)
{
[[maybe_unused]] bool matchedSwitch = false;
switch(functor.ranges[k])
{
case Range::OneNeighbourhood:
matchedSwitch = true;
for(size_t i = 0; i < TArgs::capacity; ++i)
{
args(i).arg[k] = rngWrapper.getNumber(distOne, eng);
}
break;

case Range::PositiveOnly:
matchedSwitch = true;
args(0).arg[k] = max;
for(size_t i = 1; i < TArgs::capacity; ++i)
{
args(i).arg[k] = rngWrapper.getNumber(dist, eng) + TData{1};
}
break;

case Range::PositiveAndZero:
matchedSwitch = true;
args(0).arg[k] = TData{0};
args(1).arg[k] = max;
for(size_t i = 2; i < TArgs::capacity; ++i)
{
args(i).arg[k] = rngWrapper.getNumber(dist, eng);
}
break;

case Range::NotZero:
matchedSwitch = true;
args(0).arg[k] = max;
args(1).arg[k] = low;
for(size_t i = 2; i < TArgs::capacity; ++i)
{
TData arg;
do
{
arg = rngWrapper.getNumber(dist, eng);
} while(std::equal_to<TData>()(arg, 1));
if(i % 2 == 0)
args(i).arg[k] = arg;
else
args(i).arg[k] = -arg;
}
break;

case Range::Unrestricted:
matchedSwitch = true;
args(0).arg[k] = TData{0};
args(1).arg[k] = max;
args(2).arg[k] = low;
for(size_t i = 3; i < TArgs::capacity; ++i)
{
if(i % 2 == 0)
args(i).arg[k] = rngWrapper.getNumber(dist, eng);
else
args(i).arg[k] = -rngWrapper.getNumber(dist, eng);
}
break;

case Range::Anything:
matchedSwitch = true;
args(0).arg[k] = TData{0};
args(1).arg[k] = std::numeric_limits<TData>::quiet_NaN();
args(2).arg[k] = std::numeric_limits<TData>::signaling_NaN();
args(3).arg[k] = std::numeric_limits<TData>::infinity();
args(4).arg[k] = -std::numeric_limits<TData>::infinity();
constexpr size_t nFixed = 5;
size_t i = nFixed;
for(; i < TArgs::capacity; ++i)
{
const TData v = rngWrapper.getNumber(dist, eng);
args(i).arg[k] = (i % 2 == 0) ? v : -v;
}
break;
}
assert(matchedSwitch);
}
}

} 
} 
} 
} 
