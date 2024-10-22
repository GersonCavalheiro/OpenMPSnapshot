
#ifndef BOOST_MPL_PRINT_HPP_INCLUDED
#define BOOST_MPL_PRINT_HPP_INCLUDED



#include <boost/mpl/aux_/config/msvc.hpp>
#include <boost/mpl/identity.hpp>

namespace boost { namespace mpl {

namespace aux {
#if defined(BOOST_MSVC)
# pragma warning(push, 3)
# pragma warning(disable: 4307)
#elif defined(__MWERKS__)
# pragma warn_hidevirtual on
struct print_base { virtual void f() {} };
#endif

#if defined(__EDG_VERSION__)
template <class T>
struct dependent_unsigned
{
static const unsigned value = 1;
};
#endif
} 

template <class T>
struct print
: mpl::identity<T>
#if defined(__MWERKS__)
, aux::print_base
#endif 
{
#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wc++11-extensions"
const int m_x = 1 / (sizeof(T) - sizeof(T));
# pragma clang diagnostic pop
#elif defined(BOOST_MSVC)
enum { n = sizeof(T) + -1 };
#elif defined(__MWERKS__)
void f(int);
#else 
enum {
n =
# if defined(__EDG_VERSION__)
aux::dependent_unsigned<T>::value > -1
# else 
sizeof(T) > -1
# endif 
};
#endif 
};

#if defined(BOOST_MSVC)
# pragma warning(pop)
#elif defined(__MWERKS__)
# pragma warn_hidevirtual reset
#endif

}}

#endif 
