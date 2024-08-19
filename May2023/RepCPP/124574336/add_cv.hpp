


#ifndef BOOST_TT_ADD_CV_HPP_INCLUDED
#define BOOST_TT_ADD_CV_HPP_INCLUDED

#include <boost/config.hpp>

namespace boost {


#if defined(BOOST_MSVC)
#   pragma warning(push)
#   pragma warning(disable:4181) 
#endif 

template <class T> struct add_cv{ typedef T const volatile type; };

#if defined(BOOST_MSVC)
#   pragma warning(pop)
#endif 

template <class T> struct add_cv<T&>{ typedef T& type; };

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)

template <class T> using add_cv_t = typename add_cv<T>::type;

#endif

} 

#endif 
