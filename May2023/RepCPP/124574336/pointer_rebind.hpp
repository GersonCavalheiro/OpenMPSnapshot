
#ifndef BOOST_INTRUSIVE_POINTER_REBIND_HPP
#define BOOST_INTRUSIVE_POINTER_REBIND_HPP

#ifndef BOOST_INTRUSIVE_DETAIL_WORKAROUND_HPP
#include <boost/intrusive/detail/workaround.hpp>
#endif   

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {

template <typename Ptr, typename U>
struct pointer_has_rebind
{
template <typename V> struct any
{  any(const V&) { } };

template <typename X>
static char test(int, typename X::template rebind<U>*);

template <typename X>
static int test(any<int>, void*);

static const bool value = (1 == sizeof(test<Ptr>(0, 0)));
};

template <typename Ptr, typename U>
struct pointer_has_rebind_other
{
template <typename V> struct any
{  any(const V&) { } };

template <typename X>
static char test(int, typename X::template rebind<U>::other*);

template <typename X>
static int test(any<int>, void*);

static const bool value = (1 == sizeof(test<Ptr>(0, 0)));
};

template <typename Ptr, typename U>
struct pointer_rebind_mode
{
static const unsigned int rebind =       (unsigned int)pointer_has_rebind<Ptr, U>::value;
static const unsigned int rebind_other = (unsigned int)pointer_has_rebind_other<Ptr, U>::value;
static const unsigned int mode =         rebind + rebind*rebind_other;
};

template <typename Ptr, typename U, unsigned int RebindMode>
struct pointer_rebinder;

template <typename Ptr, typename U>
struct pointer_rebinder< Ptr, U, 2u >
{
typedef typename Ptr::template rebind<U>::other type;
};

template <typename Ptr, typename U>
struct pointer_rebinder< Ptr, U, 1u >
{
typedef typename Ptr::template rebind<U> type;
};

#if !defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

template <template <class, class...> class Ptr, typename A, class... An, class U>
struct pointer_rebinder<Ptr<A, An...>, U, 0u >
{
typedef Ptr<U, An...> type;
};

template <template <class> class Ptr, typename A, class U>
struct pointer_rebinder<Ptr<A>, U, 0u >
{
typedef Ptr<U> type;
};

#else 

template <template <class> class Ptr  
, typename A
, class U>
struct pointer_rebinder<Ptr<A>, U, 0u>
{  typedef Ptr<U> type;   };

template <template <class, class> class Ptr  
, typename A, class P0
, class U>
struct pointer_rebinder<Ptr<A, P0>, U, 0u>
{  typedef Ptr<U, P0> type;   };

template <template <class, class, class> class Ptr  
, typename A, class P0, class P1
, class U>
struct pointer_rebinder<Ptr<A, P0, P1>, U, 0u>
{  typedef Ptr<U, P0, P1> type;   };

template <template <class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2>, U, 0u>
{  typedef Ptr<U, P0, P1, P2> type;   };

template <template <class, class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2, class P3
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2, P3>, U, 0u>
{  typedef Ptr<U, P0, P1, P2, P3> type;   };

template <template <class, class, class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2, class P3, class P4
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2, P3, P4>, U, 0u>
{  typedef Ptr<U, P0, P1, P2, P3, P4> type;   };

template <template <class, class, class, class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2, class P3, class P4, class P5
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2, P3, P4, P5>, U, 0u>
{  typedef Ptr<U, P0, P1, P2, P3, P4, P5> type;   };

template <template <class, class, class, class, class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2, class P3, class P4, class P5, class P6
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2, P3, P4, P5, P6>, U, 0u>
{  typedef Ptr<U, P0, P1, P2, P3, P4, P5, P6> type;   };

template <template <class, class, class, class, class, class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2, class P3, class P4, class P5, class P6, class P7
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2, P3, P4, P5, P6, P7>, U, 0u>
{  typedef Ptr<U, P0, P1, P2, P3, P4, P5, P6, P7> type;   };

template <template <class, class, class, class, class, class, class, class, class, class> class Ptr  
, typename A, class P0, class P1, class P2, class P3, class P4, class P5, class P6, class P7, class P8
, class U>
struct pointer_rebinder<Ptr<A, P0, P1, P2, P3, P4, P5, P6, P7, P8>, U, 0u>
{  typedef Ptr<U, P0, P1, P2, P3, P4, P5, P6, P7, P8> type;   };

#endif   

template <typename Ptr, typename U>
struct pointer_rebind
: public pointer_rebinder<Ptr, U, pointer_rebind_mode<Ptr, U>::mode>
{};

template <typename T, typename U>
struct pointer_rebind<T*, U>
{  typedef U* type; };

}  
}  

#endif 
