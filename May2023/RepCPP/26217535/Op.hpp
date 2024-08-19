

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>

#include <algorithm>
#include <type_traits>


namespace alpaka
{
struct AtomicAdd
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wconversion"
#endif
ref += value;
return old;
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
}
};
struct AtomicSub
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wconversion"
#endif
ref -= value;
#if BOOST_COMP_GNUC
#    pragma GCC diagnostic pop
#endif
return old;
}
};
struct AtomicMin
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref = std::min(ref, value);
return old;
}
};
struct AtomicMax
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref = std::max(ref, value);
return old;
}
};
struct AtomicExch
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref = value;
return old;
}
};
struct AtomicInc
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref = ((old >= value) ? static_cast<T>(0) : static_cast<T>(old + static_cast<T>(1)));
return old;
}
};
struct AtomicDec
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref = (((old == static_cast<T>(0)) || (old > value)) ? value : static_cast<T>(old - static_cast<T>(1)));
return old;
}
};
struct AtomicAnd
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref &= value;
return old;
}
};
struct AtomicOr
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref |= value;
return old;
}
};
struct AtomicXor
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
ALPAKA_FN_HOST_ACC auto operator()(T* const addr, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;
ref ^= value;
return old;
}
};
struct AtomicCas
{
ALPAKA_NO_HOST_ACC_WARNING
template<typename T, std::enable_if_t<!std::is_floating_point_v<T>, bool> = true>
ALPAKA_FN_HOST_ACC auto operator()(T* addr, T const& compare, T const& value) const -> T
{
auto const old = *addr;
auto& ref = *addr;

#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 4, 0))
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
ref = ((old == compare) ? value : old);
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 4, 0))
#    pragma GCC diagnostic pop
#endif
return old;
}
ALPAKA_NO_HOST_ACC_WARNING
template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
ALPAKA_FN_HOST_ACC auto operator()(T* addr, T const& compare, T const& value) const -> T
{
static_assert(sizeof(T) == 4u || sizeof(T) == 8u, "AtomicCas is supporting only 32bit and 64bit values!");
using BitType = std::conditional_t<sizeof(T) == 4u, unsigned int, unsigned long long>;

struct BitUnion
{
union
{
T value;
BitType r;
};
};

auto const old = *addr;
auto& ref = *addr;

#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 4, 0))
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wstrict-overflow"
#endif
BitUnion o{old};
BitUnion c{compare};

ref = ((o.r == c.r) ? value : old);
#if BOOST_COMP_GNUC && (BOOST_COMP_GNUC == BOOST_VERSION_NUMBER(7, 4, 0))
#    pragma GCC diagnostic pop
#endif
return old;
}
};
} 
