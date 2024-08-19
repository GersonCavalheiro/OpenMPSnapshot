

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/rand/TinyMT/Engine.hpp>
#include <alpaka/rand/Traits.hpp>

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka::rand
{
class TinyMersenneTwister : public concepts::Implements<ConceptRand, TinyMersenneTwister>
{
};
using RandStdLib = TinyMersenneTwister;

class MersenneTwister : public concepts::Implements<ConceptRand, MersenneTwister>
{
};

class RandomDevice : public concepts::Implements<ConceptRand, RandomDevice>
{
};

namespace engine::cpu
{
class MersenneTwister
{
std::mt19937 state;

public:
MersenneTwister() = default;

ALPAKA_FN_HOST MersenneTwister(
std::uint32_t const& seed,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0)
: 
state((seed ^ subsequence) + offset)
{
}

using result_type = std::mt19937::result_type;
ALPAKA_FN_HOST constexpr static auto min() -> result_type
{
return std::mt19937::min();
}
ALPAKA_FN_HOST constexpr static auto max() -> result_type
{
return std::mt19937::max();
}
ALPAKA_FN_HOST auto operator()() -> result_type
{
return state();
}
};

class TinyMersenneTwister
{
TinyMTengine state;

public:
TinyMersenneTwister() = default;

ALPAKA_FN_HOST TinyMersenneTwister(
std::uint32_t const& seed,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0)
: 
state((seed ^ subsequence) + offset)
{
}

using result_type = TinyMTengine::result_type;
ALPAKA_FN_HOST constexpr static auto min() -> result_type
{
return TinyMTengine::min();
}
ALPAKA_FN_HOST constexpr static auto max() -> result_type
{
return TinyMTengine::max();
}
ALPAKA_FN_HOST auto operator()() -> result_type
{
return state();
}
};

class RandomDevice
{
std::random_device state;

public:
RandomDevice() = default;

ALPAKA_FN_HOST RandomDevice(std::uint32_t const&, std::uint32_t const& = 0, std::uint32_t const& = 0)
{
}

using result_type = std::random_device::result_type;
ALPAKA_FN_HOST constexpr static auto min() -> result_type
{
return std::random_device::min();
}
ALPAKA_FN_HOST constexpr static auto max() -> result_type
{
return std::random_device::max();
}
ALPAKA_FN_HOST auto operator()() -> result_type
{
return state();
}
};
} 

namespace distribution::cpu
{
template<typename T>
struct NormalReal
{
template<typename TEngine>
ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
{
return m_dist(engine);
}

private:
std::normal_distribution<T> m_dist;
};

template<typename T>
struct UniformReal
{
template<typename TEngine>
ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
{
return m_dist(engine);
}

private:
std::uniform_real_distribution<T> m_dist;
};

template<typename T>
struct UniformUint
{
template<typename TEngine>
ALPAKA_FN_HOST auto operator()(TEngine& engine) -> T
{
return m_dist(engine);
}

private:
std::uniform_int_distribution<T> m_dist{
0, 
std::numeric_limits<T>::max()};
};
} 

namespace distribution::trait
{
template<typename T>
struct CreateNormalReal<RandStdLib, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
ALPAKA_FN_HOST static auto createNormalReal(RandStdLib const& ) -> cpu::NormalReal<T>
{
return {};
}
};
template<typename T>
struct CreateUniformReal<RandStdLib, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
ALPAKA_FN_HOST static auto createUniformReal(RandStdLib const& ) -> cpu::UniformReal<T>
{
return {};
}
};
template<typename T>
struct CreateUniformUint<RandStdLib, T, std::enable_if_t<std::is_integral_v<T>>>
{
ALPAKA_FN_HOST static auto createUniformUint(RandStdLib const& ) -> cpu::UniformUint<T>
{
return {};
}
};
} 

namespace engine::trait
{
template<>
struct CreateDefault<TinyMersenneTwister>
{
ALPAKA_FN_HOST static auto createDefault(
TinyMersenneTwister const& ,
std::uint32_t const& seed = 0,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0) -> cpu::TinyMersenneTwister
{
return {seed, subsequence, offset};
}
};

template<>
struct CreateDefault<MersenneTwister>
{
ALPAKA_FN_HOST static auto createDefault(
MersenneTwister const& ,
std::uint32_t const& seed = 0,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0) -> cpu::MersenneTwister
{
return {seed, subsequence, offset};
}
};

template<>
struct CreateDefault<RandomDevice>
{
ALPAKA_FN_HOST static auto createDefault(
RandomDevice const& ,
std::uint32_t const& seed = 0,
std::uint32_t const& subsequence = 0,
std::uint32_t const& offset = 0) -> cpu::RandomDevice
{
return {seed, subsequence, offset};
}
};
} 
} 
