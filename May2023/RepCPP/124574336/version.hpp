#ifndef BOOST_SERIALIZATION_VERSION_HPP
#define BOOST_SERIALIZATION_VERSION_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/integral_c_tag.hpp>

#include <boost/type_traits/is_base_and_derived.hpp>

namespace boost {
namespace serialization {

struct basic_traits;

template<class T>
struct version
{
template<class U>
struct traits_class_version {
typedef typename U::version type;
};

typedef mpl::integral_c_tag tag;
typedef
typename mpl::eval_if<
is_base_and_derived<boost::serialization::basic_traits,T>,
traits_class_version< T >,
mpl::int_<0>
>::type type;
BOOST_STATIC_CONSTANT(int, value = version::type::value);
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<class T>
const int version<T>::value;
#endif

} 
} 



#include <boost/mpl/less.hpp>
#include <boost/mpl/comparison.hpp>

#define BOOST_CLASS_VERSION(T, N)                                      \
namespace boost {                                                      \
namespace serialization {                                              \
template<>                                                             \
struct version<T >                                                     \
{                                                                      \
typedef mpl::int_<N> type;                                         \
typedef mpl::integral_c_tag tag;                                   \
BOOST_STATIC_CONSTANT(int, value = version::type::value);          \
BOOST_MPL_ASSERT((                                                 \
boost::mpl::less<                                              \
boost::mpl::int_<N>,                                       \
boost::mpl::int_<256>                                      \
>                                                              \
));                                                                \
\
};                                                                     \
}                                                                      \
}

#endif 
