

#ifndef BOOST_TT_ADD_VOLATILE_HPP_INCLUDED
#define BOOST_TT_ADD_VOLATILE_HPP_INCLUDED

#include <boost/config.hpp>

namespace boost {


#if defined(BOOST_MSVC)
#   pragma warning(push)
#   pragma warning(disable:4181) 
#endif 

template <class T> struct add_volatile{ typedef T volatile type; };

#if defined(BOOST_MSVC)
#   pragma warning(pop)
#endif 

template <class T> struct add_volatile<T&>{ typedef T& type; };

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)

template <class T> using add_volatile_t = typename add_volatile<T>::type;

#endif

} 

#endif 
