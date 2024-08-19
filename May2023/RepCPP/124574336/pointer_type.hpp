
#ifndef BOOST_INTERPROCESS_DETAIL_POINTER_TYPE_HPP
#define BOOST_INTERPROCESS_DETAIL_POINTER_TYPE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>
#include <boost/interprocess/detail/type_traits.hpp>

namespace boost {
namespace interprocess {
namespace ipcdetail {

struct two {char _[2];};

namespace pointer_type_imp {

template <class U> static two  test(...);
template <class U> static char test(typename U::pointer* = 0);

}  

template <class T>
struct has_pointer_type
{
static const bool value = sizeof(pointer_type_imp::test<T>(0)) == 1;
};

namespace pointer_type_imp {

template <class T, class D, bool = has_pointer_type<D>::value>
struct pointer_type
{
typedef typename D::pointer type;
};

template <class T, class D>
struct pointer_type<T, D, false>
{
typedef T* type;
};

}  

template <class T, class D>
struct pointer_type
{
typedef typename pointer_type_imp::pointer_type<T,
typename remove_reference<D>::type>::type type;
};

}  
}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

