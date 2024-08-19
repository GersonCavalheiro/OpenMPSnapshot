



#ifndef BOOST_STATIC_ASSERT_HPP
#define BOOST_STATIC_ASSERT_HPP

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if defined(__GNUC__) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#pragma GCC system_header
#endif

#ifndef BOOST_NO_CXX11_STATIC_ASSERT
#  ifndef BOOST_NO_CXX11_VARIADIC_MACROS
#     define BOOST_STATIC_ASSERT_MSG( ... ) static_assert(__VA_ARGS__)
#  else
#     define BOOST_STATIC_ASSERT_MSG( B, Msg ) static_assert( B, Msg )
#  endif
#else
#     define BOOST_STATIC_ASSERT_MSG( B, Msg ) BOOST_STATIC_ASSERT( B )
#endif

#ifdef BOOST_BORLANDC
#define BOOST_BUGGY_INTEGRAL_CONSTANT_EXPRESSIONS
#endif

#if defined(__GNUC__) && (__GNUC__ == 3) && ((__GNUC_MINOR__ == 3) || (__GNUC_MINOR__ == 4))
#  define BOOST_SA_GCC_WORKAROUND
#endif

#if defined(__GNUC__) && ((__GNUC__ > 3) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 4)))
#  ifndef BOOST_NO_CXX11_VARIADIC_MACROS
#     define BOOST_STATIC_ASSERT_BOOL_CAST( ... ) ((__VA_ARGS__) != 0)
#  else
#     define BOOST_STATIC_ASSERT_BOOL_CAST( x ) ((x) != 0)
#  endif
#else
#  ifndef BOOST_NO_CXX11_VARIADIC_MACROS
#     define BOOST_STATIC_ASSERT_BOOL_CAST( ... ) (bool)(__VA_ARGS__)
#  else
#     define BOOST_STATIC_ASSERT_BOOL_CAST(x) (bool)(x)
#  endif
#endif

#ifndef BOOST_NO_CXX11_STATIC_ASSERT
#  ifndef BOOST_NO_CXX11_VARIADIC_MACROS
#     define BOOST_STATIC_ASSERT( ... ) static_assert(__VA_ARGS__, #__VA_ARGS__)
#  else
#     define BOOST_STATIC_ASSERT( B ) static_assert(B, #B)
#  endif
#else

namespace boost{

template <bool x> struct STATIC_ASSERTION_FAILURE;

template <> struct STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };

template<int x> struct static_assert_test{};

}

#if !defined(BOOST_BUGGY_INTEGRAL_CONSTANT_EXPRESSIONS)

#if defined(BOOST_MSVC) && defined(BOOST_NO_CXX11_VARIADIC_MACROS)
#define BOOST_STATIC_ASSERT( B ) \
typedef ::boost::static_assert_test<\
sizeof(::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST ( B ) >)>\
BOOST_JOIN(boost_static_assert_typedef_, __COUNTER__)
#elif defined(BOOST_MSVC)
#define BOOST_STATIC_ASSERT(...) \
typedef ::boost::static_assert_test<\
sizeof(::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST (__VA_ARGS__) >)>\
BOOST_JOIN(boost_static_assert_typedef_, __COUNTER__)
#elif (defined(BOOST_INTEL_CXX_VERSION) || defined(BOOST_SA_GCC_WORKAROUND))  && defined(BOOST_NO_CXX11_VARIADIC_MACROS)
# define BOOST_STATIC_ASSERT( B ) \
typedef char BOOST_JOIN(boost_static_assert_typedef_, __LINE__) \
[ ::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST( B ) >::value ]
#elif (defined(BOOST_INTEL_CXX_VERSION) || defined(BOOST_SA_GCC_WORKAROUND))  && !defined(BOOST_NO_CXX11_VARIADIC_MACROS)
# define BOOST_STATIC_ASSERT(...) \
typedef char BOOST_JOIN(boost_static_assert_typedef_, __LINE__) \
[ ::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST( __VA_ARGS__ ) >::value ]
#elif defined(__sgi)
#define BOOST_STATIC_ASSERT( B ) \
BOOST_STATIC_CONSTANT(bool, \
BOOST_JOIN(boost_static_assert_test_, __LINE__) = ( B )); \
typedef ::boost::static_assert_test<\
sizeof(::boost::STATIC_ASSERTION_FAILURE< \
BOOST_JOIN(boost_static_assert_test_, __LINE__) >)>\
BOOST_JOIN(boost_static_assert_typedef_, __LINE__)
#elif BOOST_WORKAROUND(__MWERKS__, <= 0x3003)
#define BOOST_STATIC_ASSERT( B ) \
BOOST_STATIC_CONSTANT(int, \
BOOST_JOIN(boost_static_assert_test_, __LINE__) = \
sizeof(::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST( B ) >) )
#else
#  ifndef BOOST_NO_CXX11_VARIADIC_MACROS
#     define BOOST_STATIC_ASSERT( ... ) \
typedef ::boost::static_assert_test<\
sizeof(::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST( __VA_ARGS__ ) >)>\
BOOST_JOIN(boost_static_assert_typedef_, __LINE__) BOOST_ATTRIBUTE_UNUSED
#  else
#     define BOOST_STATIC_ASSERT( B ) \
typedef ::boost::static_assert_test<\
sizeof(::boost::STATIC_ASSERTION_FAILURE< BOOST_STATIC_ASSERT_BOOL_CAST( B ) >)>\
BOOST_JOIN(boost_static_assert_typedef_, __LINE__) BOOST_ATTRIBUTE_UNUSED
#  endif
#endif

#else
#  ifndef BOOST_NO_CXX11_VARIADIC_MACROS
#    define BOOST_STATIC_ASSERT( ... ) \
enum { BOOST_JOIN(boost_static_assert_enum_, __LINE__) \
= sizeof(::boost::STATIC_ASSERTION_FAILURE< (bool)( __VA_ARGS__ ) >) }
#  else
#    define BOOST_STATIC_ASSERT(B) \
enum { BOOST_JOIN(boost_static_assert_enum_, __LINE__) \
= sizeof(::boost::STATIC_ASSERTION_FAILURE< (bool)( B ) >) }
#  endif
#endif
#endif 

#endif 


