#ifndef BOOST_SERIALIZATION_COLLECTION_TRAITS_HPP
#define BOOST_SERIALIZATION_COLLECTION_TRAITS_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/mpl/integral_c_tag.hpp>

#include <boost/cstdint.hpp>
#include <boost/integer_traits.hpp>
#include <climits> 
#include <boost/serialization/level.hpp>

#define BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(T, C)          \
template<>                                                          \
struct implementation_level< C < T > > {                            \
typedef mpl::integral_c_tag tag;                                \
typedef mpl::int_<object_serializable> type;                    \
BOOST_STATIC_CONSTANT(int, value = object_serializable);        \
};                                                                  \


#if defined(BOOST_NO_CWCHAR) || defined(BOOST_NO_INTRINSIC_WCHAR_T)
#define BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER_WCHAR(C)
#else
#define BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER_WCHAR(C)   \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(wchar_t, C)        \

#endif

#if defined(BOOST_HAS_LONG_LONG)
#define BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER_INT64(C)    \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(boost::long_long_type, C)  \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(boost::ulong_long_type, C) \

#else
#define BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER_INT64(C)
#endif

#define BOOST_SERIALIZATION_COLLECTION_TRAITS(C)                     \
namespace boost { namespace serialization {                      \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(bool, C)            \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(char, C)            \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(signed char, C)     \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(unsigned char, C)   \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(signed int, C)      \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(unsigned int, C)    \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(signed long, C)     \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(unsigned long, C)   \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(float, C)           \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(double, C)          \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(unsigned short, C)  \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER(signed short, C)    \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER_INT64(C)            \
BOOST_SERIALIZATION_COLLECTION_TRAITS_HELPER_WCHAR(C)            \
} }                                                              \


#endif 
