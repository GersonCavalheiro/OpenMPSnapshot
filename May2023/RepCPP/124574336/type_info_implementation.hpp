#ifndef BOOST_SERIALIZATION_TYPE_INFO_IMPLEMENTATION_HPP
#define BOOST_SERIALIZATION_TYPE_INFO_IMPLEMENTATION_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/static_assert.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>
#include <boost/serialization/traits.hpp>

namespace boost {
namespace serialization {

template<class T>
struct type_info_implementation {
template<class U>
struct traits_class_typeinfo_implementation {
typedef typename U::type_info_implementation::type type;
};
typedef
typename mpl::eval_if<
is_base_and_derived<boost::serialization::basic_traits, T>,
traits_class_typeinfo_implementation< T >,
mpl::identity<
typename extended_type_info_impl< T >::type
>
>::type type;
};

} 
} 

#define BOOST_CLASS_TYPE_INFO(T, ETI)              \
namespace boost {                                  \
namespace serialization {                          \
template<>                                         \
struct type_info_implementation< T > {             \
typedef ETI type;                              \
};                                                 \
template<>                                         \
struct type_info_implementation< const T > {       \
typedef ETI type;                              \
};                                                 \
}                                                  \
}                                                  \


#endif 
