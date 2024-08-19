

#ifndef BOOST_TT_ADD_CONST_HPP_INCLUDED
#define BOOST_TT_ADD_CONST_HPP_INCLUDED

#include <boost/type_traits/detail/config.hpp>

namespace boost {


#if defined(BOOST_MSVC)
#   pragma warning(push)
#   pragma warning(disable:4181) 
#endif 

template <class T> struct add_const
{
typedef T const type;
};

#if defined(BOOST_MSVC)
#   pragma warning(pop)
#endif 

template <class T> struct add_const<T&>
{
typedef T& type;
};

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)

template <class T> using add_const_t = typename add_const<T>::type;

#endif

} 

#endif 
