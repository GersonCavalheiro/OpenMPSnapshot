

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/math/Traits.hpp>
#include <alpaka/rand/RandPhilox.hpp>
#include <alpaka/rand/Traits.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace alpaka::rand
{
class RandDefault : public concepts::Implements<ConceptRand, RandDefault>
{
};

namespace distribution::gpu
{
namespace detail
{
template<typename TFloat>
struct BitsType;

template<>
struct BitsType<float>
{
using type = std::uint32_t;
};
template<>
struct BitsType<double>
{
using type = std::uint64_t;
};
} 

template<typename T>
class UniformUint
{
static_assert(std::is_integral_v<T>, "Return type of UniformUint must be integral.");

public:
UniformUint() = default;

template<typename TEngine>
ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
{
using BitsT = typename TEngine::result_type;
T ret = 0;
constexpr auto N = sizeof(T) / sizeof(BitsT);
for(unsigned int a = 0; a < N; ++a)
{
ret
^= (static_cast<T>(engine())
<< (sizeof(BitsT) * std::numeric_limits<unsigned char>::digits * a));
}
return ret;
}
};

template<typename T>
class UniformReal
{
static_assert(std::is_floating_point_v<T>, "Return type of UniformReal must be floating point.");

using BitsT = typename detail::BitsType<T>::type;

public:
UniformReal() = default;

template<typename TEngine>
ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
{
constexpr BitsT limit = static_cast<BitsT>(1) << std::numeric_limits<T>::digits;
const BitsT b = UniformUint<BitsT>()(engine);
auto const ret = static_cast<T>(b & (limit - 1)) / limit;
return ret;
}
};


template<typename Acc, typename T>
class NormalReal
{
static_assert(std::is_floating_point_v<T>, "Return type of NormalReal must be floating point.");

Acc const* m_acc;
T m_cache = std::numeric_limits<T>::quiet_NaN();

public:

ALPAKA_FN_HOST_ACC constexpr NormalReal(Acc const& acc) : m_acc(&acc)
{
}


ALPAKA_FN_HOST_ACC constexpr NormalReal(NormalReal const& other) : m_acc(other.m_acc)
{
}

ALPAKA_FN_HOST_ACC constexpr auto operator=(NormalReal const& other) -> NormalReal&
{
m_acc = other.m_acc;
return *this;
}

template<typename TEngine>
ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
{
constexpr auto sigma = T{1};
constexpr auto mu = T{0};
if(math::isnan(*m_acc, m_cache))
{
UniformReal<T> uni;

T u1, u2;
do
{
u1 = uni(engine);
u2 = uni(engine);
} while(u1 <= std::numeric_limits<T>::epsilon());

const T mag = sigma * math::sqrt(*m_acc, static_cast<T>(-2.) * math::log(*m_acc, u1));
constexpr T twoPi = static_cast<T>(2. * math::constants::pi);
m_cache = mag * static_cast<T>(math::cos(*m_acc, twoPi * u2)) + mu;

return mag * static_cast<T>(math::sin(*m_acc, twoPi * u2)) + mu;
}

const T ret = m_cache;
m_cache = std::numeric_limits<T>::quiet_NaN();
return ret;
}
};
} 

namespace distribution::trait
{
template<typename T>
struct CreateNormalReal<RandDefault, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
template<typename TAcc>
ALPAKA_FN_HOST_ACC static auto createNormalReal(TAcc const& acc) -> gpu::NormalReal<TAcc, T>
{
return {acc};
}
};
template<typename T>
struct CreateUniformReal<RandDefault, T, std::enable_if_t<std::is_floating_point_v<T>>>
{
ALPAKA_FN_HOST_ACC static auto createUniformReal(RandDefault const& ) -> gpu::UniformReal<T>
{
return {};
}
};
template<typename T>
struct CreateUniformUint<RandDefault, T, std::enable_if_t<std::is_integral_v<T>>>
{
ALPAKA_FN_HOST_ACC static auto createUniformUint(RandDefault const& ) -> gpu::UniformUint<T>
{
return {};
}
};
} 

namespace engine::trait
{
template<>
struct CreateDefault<RandDefault>
{
template<typename TAcc>
ALPAKA_FN_HOST_ACC static auto createDefault(
TAcc const& ,
std::uint32_t const& seed,
std::uint32_t const& subsequence,
std::uint32_t const& offset) -> Philox4x32x10<TAcc>
{
return {seed, subsequence, offset};
}
};
} 
} 
