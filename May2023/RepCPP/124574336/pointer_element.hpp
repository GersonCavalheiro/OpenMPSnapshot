
#ifndef BOOST_MOVE_DETAIL_POINTER_ELEMENT_HPP
#define BOOST_MOVE_DETAIL_POINTER_ELEMENT_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#ifndef BOOST_MOVE_DETAIL_WORKAROUND_HPP
#include <boost/move/detail/workaround.hpp>
#endif   

namespace boost {
namespace movelib {
namespace detail{


template <typename T> struct first_param
{  typedef void type;   };

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

template <template <typename, typename...> class TemplateClass, typename T, typename... Args>
struct first_param< TemplateClass<T, Args...> >
{
typedef T type;
};

#else 

template < template  
<class
> class TemplateClass, class T
>
struct first_param
< TemplateClass<T> >
{  typedef T type;   };

template < template  
<class,class
> class TemplateClass, class T
, class P0>
struct first_param
< TemplateClass<T, P0> >
{  typedef T type;   };

template < template  
<class,class,class
> class TemplateClass, class T
, class P0, class P1>
struct first_param
< TemplateClass<T, P0, P1> >
{  typedef T type;   };

template < template  
<class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2>
struct first_param
< TemplateClass<T, P0, P1, P2> >
{  typedef T type;   };

template < template  
<class,class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2, class P3>
struct first_param
< TemplateClass<T, P0, P1, P2, P3> >
{  typedef T type;   };

template < template  
<class,class,class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2, class P3, class P4>
struct first_param
< TemplateClass<T, P0, P1, P2, P3, P4> >
{  typedef T type;   };

template < template  
<class,class,class,class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2, class P3, class P4, class P5>
struct first_param
< TemplateClass<T, P0, P1, P2, P3, P4, P5> >
{  typedef T type;   };

template < template  
<class,class,class,class,class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2, class P3, class P4, class P5, class P6>
struct first_param
< TemplateClass<T, P0, P1, P2, P3, P4, P5, P6> >
{  typedef T type;   };

template < template  
<class,class,class,class,class,class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2, class P3, class P4, class P5, class P6, class P7>
struct first_param
< TemplateClass<T, P0, P1, P2, P3, P4, P5, P6, P7> >
{  typedef T type;   };

template < template  
<class,class,class,class,class,class,class,class,class,class
> class TemplateClass, class T
, class P0, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8>
struct first_param
< TemplateClass<T, P0, P1, P2, P3, P4, P5, P6, P7, P8> >
{  typedef T type;   };

#endif   

template <typename T>
struct has_internal_pointer_element
{
template <typename X>
static char test(int, typename X::element_type*);

template <typename X>
static int test(...);

static const bool value = (1 == sizeof(test<T>(0, 0)));
};

template<class Ptr, bool = has_internal_pointer_element<Ptr>::value>
struct pointer_element_impl
{
typedef typename Ptr::element_type type;
};

template<class Ptr>
struct pointer_element_impl<Ptr, false>
{
typedef typename boost::movelib::detail::first_param<Ptr>::type type;
};

}  

template <typename Ptr>
struct pointer_element
{
typedef typename ::boost::movelib::detail::pointer_element_impl<Ptr>::type type;
};

template <typename T>
struct pointer_element<T*>
{  typedef T type; };

}  
}  

#endif 
