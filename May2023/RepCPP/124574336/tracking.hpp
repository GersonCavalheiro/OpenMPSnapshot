#ifndef BOOST_SERIALIZATION_TRACKING_HPP
#define BOOST_SERIALIZATION_TRACKING_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/greater.hpp>
#include <boost/mpl/integral_c_tag.hpp>

#include <boost/type_traits/is_base_and_derived.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/tracking_enum.hpp>
#include <boost/serialization/type_info_implementation.hpp>

namespace boost {
namespace serialization {

struct basic_traits;

template<class T>
struct tracking_level_impl {
template<class U>
struct traits_class_tracking {
typedef typename U::tracking type;
};
typedef mpl::integral_c_tag tag;
typedef
typename mpl::eval_if<
is_base_and_derived<boost::serialization::basic_traits, T>,
traits_class_tracking< T >,
typename mpl::eval_if<
is_pointer< T >,
mpl::int_<track_never>,
typename mpl::eval_if<
typename mpl::equal_to<
implementation_level< T >,
mpl::int_<primitive_type>
>,
mpl::int_<track_never>,
mpl::int_<track_selectively>
>  > >::type type;
BOOST_STATIC_CONSTANT(int, value = type::value);
};

template<class T>
struct tracking_level :
public tracking_level_impl<const T>
{
};

template<class T, enum tracking_type L>
inline bool operator>=(tracking_level< T > t, enum tracking_type l)
{
return t.value >= (int)l;
}

} 
} 


#define BOOST_CLASS_TRACKING(T, E)           \
namespace boost {                            \
namespace serialization {                    \
template<>                                   \
struct tracking_level< T >                   \
{                                            \
typedef mpl::integral_c_tag tag;         \
typedef mpl::int_< E> type;              \
BOOST_STATIC_CONSTANT(                   \
int,                                 \
value = tracking_level::type::value  \
);                                       \
\
BOOST_STATIC_ASSERT((                    \
mpl::greater<                        \
\
implementation_level< T >,       \
mpl::int_<primitive_type>        \
>::value                             \
));                                      \
};                                           \
}}

#endif 
