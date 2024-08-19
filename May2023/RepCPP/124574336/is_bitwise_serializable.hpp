





#ifndef BOOST_SERIALIZATION_IS_BITWISE_SERIALIZABLE_HPP
#define BOOST_SERIALIZATION_IS_BITWISE_SERIALIZABLE_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/bool_fwd.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

namespace boost {
namespace serialization {
template<class T>
struct is_bitwise_serializable
: public is_arithmetic< T >
{};
} 
} 


#define BOOST_IS_BITWISE_SERIALIZABLE(T)              \
namespace boost {                                     \
namespace serialization {                             \
template<>                                            \
struct is_bitwise_serializable< T > : mpl::true_ {};  \
}}                                                    \


#endif 
