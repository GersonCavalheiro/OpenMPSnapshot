#ifndef BOOST_SERIALIZATION_SMART_CAST_HPP
#define BOOST_SERIALIZATION_SMART_CAST_HPP

#if defined(_MSC_VER)
# pragma once
#endif









#include <exception>
#include <typeinfo>
#include <cstddef> 

#include <boost/config.hpp>
#include <boost/static_assert.hpp>

#include <boost/type_traits/is_base_and_derived.hpp>
#include <boost/type_traits/is_polymorphic.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_reference.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/identity.hpp>

#include <boost/serialization/throw_exception.hpp>

namespace boost {
namespace serialization {
namespace smart_cast_impl {

template<class T>
struct reference {

struct polymorphic {

struct linear {
template<class U>
static T cast(U & u){
return static_cast< T >(u);
}
};

struct cross {
template<class U>
static T cast(U & u){
return dynamic_cast< T >(u);
}
};

template<class U>
static T cast(U & u){
#if ! defined(NDEBUG)                               \
|| defined(__MWERKS__)
return cross::cast(u);
#else
typedef typename mpl::eval_if<
typename mpl::and_<
mpl::not_<is_base_and_derived<
typename remove_reference< T >::type,
U
> >,
mpl::not_<is_base_and_derived<
U,
typename remove_reference< T >::type
> >
>,
mpl::identity<cross>,
mpl::identity<linear>
>::type typex;
return typex::cast(u);
#endif
}
};

struct non_polymorphic {
template<class U>
static T cast(U & u){
return static_cast< T >(u);
}
};
template<class U>
static T cast(U & u){
typedef typename mpl::eval_if<
boost::is_polymorphic<U>,
mpl::identity<polymorphic>,
mpl::identity<non_polymorphic>
>::type typex;
return typex::cast(u);
}
};

template<class T>
struct pointer {

struct polymorphic {
#if 0
struct linear {
template<class U>
static T cast(U * u){
return static_cast< T >(u);
}
};

struct cross {
template<class U>
static T cast(U * u){
T tmp = dynamic_cast< T >(u);
#ifndef NDEBUG
if ( tmp == 0 ) throw_exception(std::bad_cast());
#endif
return tmp;
}
};

template<class U>
static T cast(U * u){
typedef
typename mpl::eval_if<
typename mpl::and_<
mpl::not_<is_base_and_derived<
typename remove_pointer< T >::type,
U
> >,
mpl::not_<is_base_and_derived<
U,
typename remove_pointer< T >::type
> >
>,
mpl::identity<cross>,
mpl::identity<linear>
>::type typex;
return typex::cast(u);
}
#else
template<class U>
static T cast(U * u){
T tmp = dynamic_cast< T >(u);
#ifndef NDEBUG
if ( tmp == 0 ) throw_exception(std::bad_cast());
#endif
return tmp;
}
#endif
};

struct non_polymorphic {
template<class U>
static T cast(U * u){
return static_cast< T >(u);
}
};

template<class U>
static T cast(U * u){
typedef typename mpl::eval_if<
boost::is_polymorphic<U>,
mpl::identity<polymorphic>,
mpl::identity<non_polymorphic>
>::type typex;
return typex::cast(u);
}

};

template<class TPtr>
struct void_pointer {
template<class UPtr>
static TPtr cast(UPtr uptr){
return static_cast<TPtr>(uptr);
}
};

template<class T>
struct error {
template<class U>
static T cast(U){
BOOST_STATIC_ASSERT(sizeof(T)==0);
return * static_cast<T *>(NULL);
}
};

} 

template<class T, class U>
T smart_cast(U u) {
typedef
typename mpl::eval_if<
typename mpl::or_<
boost::is_same<void *, U>,
boost::is_same<void *, T>,
boost::is_same<const void *, U>,
boost::is_same<const void *, T>
>,
mpl::identity<smart_cast_impl::void_pointer< T > >,
typename mpl::eval_if<boost::is_pointer<U>,
mpl::identity<smart_cast_impl::pointer< T > >,
typename mpl::eval_if<boost::is_reference<U>,
mpl::identity<smart_cast_impl::reference< T > >,
mpl::identity<smart_cast_impl::error< T >
>
>
>
>::type typex;
return typex::cast(u);
}

template<class T, class U>
T smart_cast_reference(U & u) {
return smart_cast_impl::reference< T >::cast(u);
}

} 
} 

#endif 
