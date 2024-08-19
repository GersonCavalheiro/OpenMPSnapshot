

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/meta/IsArrayOrVector.hpp>
#include <alpaka/rand/Philox/PhiloxSingle.hpp>
#include <alpaka/rand/Philox/PhiloxVector.hpp>
#include <alpaka/rand/Traits.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka::rand
{

template<typename TAcc>
class Philox4x32x10 : public concepts::Implements<ConceptRand, Philox4x32x10<TAcc>>
{
public:
using EngineParams = engine::PhiloxParams<4, 32, 10>; 
using EngineVariant = engine::PhiloxSingle<TAcc, EngineParams>; 


ALPAKA_FN_HOST_ACC Philox4x32x10(
std::uint64_t const seed = 0,
std::uint64_t const subsequence = 0,
std::uint64_t const offset = 0)
: engineVariant(seed, subsequence, offset)
{
}

using result_type = std::uint32_t;
ALPAKA_FN_HOST_ACC constexpr auto min() -> result_type
{
return 0;
}
ALPAKA_FN_HOST_ACC constexpr auto max() -> result_type
{
return std::numeric_limits<result_type>::max();
}
ALPAKA_FN_HOST_ACC auto operator()() -> result_type
{
return engineVariant();
}

private:
EngineVariant engineVariant;
};


template<typename TAcc>
class Philox4x32x10Vector : public concepts::Implements<ConceptRand, Philox4x32x10Vector<TAcc>>
{
public:
using EngineParams = engine::PhiloxParams<4, 32, 10>;
using EngineVariant = engine::PhiloxVector<TAcc, EngineParams>;


ALPAKA_FN_HOST_ACC Philox4x32x10Vector(
std::uint32_t const seed = 0,
std::uint32_t const subsequence = 0,
std::uint32_t const offset = 0)
: engineVariant(seed, subsequence, offset)
{
}

template<typename TScalar>
using ResultContainer = typename EngineVariant::template ResultContainer<TScalar>;

using ResultInt = std::uint32_t;
using ResultVec = decltype(std::declval<EngineVariant>()());
ALPAKA_FN_HOST_ACC constexpr auto min() -> ResultInt
{
return 0;
}
ALPAKA_FN_HOST_ACC constexpr auto max() -> ResultInt
{
return std::numeric_limits<ResultInt>::max();
}
ALPAKA_FN_HOST_ACC auto operator()() -> ResultVec
{
return engineVariant();
}

private:
EngineVariant engineVariant;
};

template<typename TEngine>
struct EngineCallHostAccProxy
{
ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> decltype(engine())
{
return engine();
}
};

template<typename TResult, typename TSfinae = void>
class UniformReal : public concepts::Implements<ConceptRand, UniformReal<TResult>>
{
template<typename TRes, typename TEnable = void>
struct ResultType
{
using type = TRes;
};

template<typename TRes>
struct ResultType<TRes, std::enable_if_t<meta::IsArrayOrVector<TRes>::value>>
{
using type = typename TRes::value_type;
};

using T = typename ResultType<TResult>::type;
static_assert(std::is_floating_point_v<T>, "Only floating-point types are supported");

public:
ALPAKA_FN_HOST_ACC UniformReal() : UniformReal(0, 1)
{
}

ALPAKA_FN_HOST_ACC UniformReal(T min, T max) : _min(min), _max(max), _range(_max - _min)
{
}

template<typename TEngine>
ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> TResult
{
if constexpr(meta::IsArrayOrVector<TResult>::value)
{
auto result = engine();
T scale = static_cast<T>(1) / engine.max() * _range;
TResult ret{
static_cast<T>(result[0]) * scale + _min,
static_cast<T>(result[1]) * scale + _min,
static_cast<T>(result[2]) * scale + _min,
static_cast<T>(result[3]) * scale + _min};
return ret;
}
else
{
return static_cast<T>(EngineCallHostAccProxy<TEngine>{}(engine)) / engine.max() * _range + _min;
}

ALPAKA_UNREACHABLE(TResult{});
}

private:
const T _min;
const T _max;
const T _range;
};
} 
