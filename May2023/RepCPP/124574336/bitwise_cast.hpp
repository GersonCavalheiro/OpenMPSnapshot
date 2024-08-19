


#ifndef BOOST_ATOMIC_DETAIL_BITWISE_CAST_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_BITWISE_CAST_HPP_INCLUDED_

#include <cstddef>
#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/addressof.hpp>
#include <boost/atomic/detail/string_ops.hpp>
#include <boost/atomic/detail/type_traits/integral_constant.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(BOOST_GCC) && BOOST_GCC >= 80000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif

namespace boost {
namespace atomics {
namespace detail {

template< std::size_t FromSize, typename To >
BOOST_FORCEINLINE void clear_tail_padding_bits(To& to, atomics::detail::true_type) BOOST_NOEXCEPT
{
BOOST_ATOMIC_DETAIL_MEMSET(reinterpret_cast< unsigned char* >(atomics::detail::addressof(to)) + FromSize, 0, sizeof(To) - FromSize);
}

template< std::size_t FromSize, typename To >
BOOST_FORCEINLINE void clear_tail_padding_bits(To&, atomics::detail::false_type) BOOST_NOEXCEPT
{
}

template< std::size_t FromSize, typename To >
BOOST_FORCEINLINE void clear_tail_padding_bits(To& to) BOOST_NOEXCEPT
{
atomics::detail::clear_tail_padding_bits< FromSize >(to, atomics::detail::integral_constant< bool, FromSize < sizeof(To) >());
}

template< typename To, std::size_t FromSize, typename From >
BOOST_FORCEINLINE To bitwise_cast(From const& from) BOOST_NOEXCEPT
{
To to;
BOOST_ATOMIC_DETAIL_MEMCPY
(
atomics::detail::addressof(to),
atomics::detail::addressof(from),
(FromSize < sizeof(To) ? FromSize : sizeof(To))
);
atomics::detail::clear_tail_padding_bits< FromSize >(to);
return to;
}

template< typename To, typename From >
BOOST_FORCEINLINE To bitwise_cast(From const& from) BOOST_NOEXCEPT
{
return atomics::detail::bitwise_cast< To, sizeof(From) >(from);
}

} 
} 
} 

#if defined(BOOST_GCC) && BOOST_GCC >= 80000
#pragma GCC diagnostic pop
#endif

#include <boost/atomic/detail/footer.hpp>

#endif 
