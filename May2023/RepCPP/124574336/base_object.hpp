#ifndef BOOST_SERIALIZATION_BASE_OBJECT_HPP
#define BOOST_SERIALIZATION_BASE_OBJECT_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/identity.hpp>

#include <boost/type_traits/is_base_and_derived.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/is_polymorphic.hpp>

#include <boost/static_assert.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/force_include.hpp>
#include <boost/serialization/void_cast_fwd.hpp>

namespace boost {
namespace serialization {

namespace detail
{
template<class B, class D>
struct base_cast
{
typedef typename
mpl::if_<
is_const<D>,
const B,
B
>::type type;
BOOST_STATIC_ASSERT(is_const<type>::value == is_const<D>::value);
};

template<class Base, class Derived>
struct base_register
{
struct polymorphic {
static void const * invoke(){
Base const * const b = 0;
Derived const * const d = 0;
return & void_cast_register(d, b);
}
};
struct non_polymorphic {
static void const * invoke(){
return 0;
}
};
static void const * invoke(){
typedef typename mpl::eval_if<
is_polymorphic<Base>,
mpl::identity<polymorphic>,
mpl::identity<non_polymorphic>
>::type type;
return type::invoke();
}
};

} 
template<class Base, class Derived>
typename detail::base_cast<Base, Derived>::type &
base_object(Derived &d)
{
BOOST_STATIC_ASSERT(( is_base_and_derived<Base,Derived>::value));
BOOST_STATIC_ASSERT(! is_pointer<Derived>::value);
typedef typename detail::base_cast<Base, Derived>::type type;
detail::base_register<type, Derived>::invoke();
return access::cast_reference<type, Derived>(d);
}

} 
} 

#endif 
