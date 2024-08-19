#ifndef BOOST_SERIALIZATION_LEVEL_HPP
#define BOOST_SERIALIZATION_LEVEL_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/type_traits/is_fundamental.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <boost/type_traits/is_array.hpp>
#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/integral_c_tag.hpp>

#include <boost/serialization/level_enum.hpp>

namespace boost {
namespace serialization {

struct basic_traits;

template<class T>
struct implementation_level_impl {
template<class U>
struct traits_class_level {
typedef typename U::level type;
};

typedef mpl::integral_c_tag tag;
typedef
typename mpl::eval_if<
is_base_and_derived<boost::serialization::basic_traits, T>,
traits_class_level< T >,
typename mpl::eval_if<
is_fundamental< T >,
mpl::int_<primitive_type>,
typename mpl::eval_if<
is_class< T >,
mpl::int_<object_class_info>,
typename mpl::eval_if<
is_array< T >,
mpl::int_<object_serializable>,
typename mpl::eval_if<
is_enum< T >,
mpl::int_<primitive_type>,
mpl::int_<not_serializable>
>
>
>
>
>::type type;
BOOST_STATIC_CONSTANT(int, value = type::value);
};

template<class T>
struct implementation_level :
public implementation_level_impl<const T>
{
};

template<class T, int L>
inline bool operator>=(implementation_level< T > t, enum level_type l)
{
return t.value >= (int)l;
}

} 
} 

#define BOOST_CLASS_IMPLEMENTATION(T, E)                 \
namespace boost {                                    \
namespace serialization {                            \
template <>                                          \
struct implementation_level_impl< const T >                     \
{                                                    \
typedef mpl::integral_c_tag tag;                 \
typedef mpl::int_< E > type;                     \
BOOST_STATIC_CONSTANT(                           \
int,                                         \
value = implementation_level_impl::type::value    \
);                                               \
};                                                   \
}                                                    \
}


#endif 
