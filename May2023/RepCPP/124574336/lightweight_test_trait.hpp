#ifndef BOOST_CORE_LIGHTWEIGHT_TEST_TRAIT_HPP
#define BOOST_CORE_LIGHTWEIGHT_TEST_TRAIT_HPP


#if defined(_MSC_VER)
# pragma once
#endif


#include <boost/core/lightweight_test.hpp>
#include <boost/core/typeinfo.hpp>
#include <boost/core/is_same.hpp>
#include <boost/config.hpp>

namespace boost
{

namespace detail
{

template<class, int = 0> struct test_print { };

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<T, 2>)
{
return o << boost::core::demangled_name(BOOST_CORE_TYPEID(T));
}

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<T, 1>)
{
return o << test_print<T, 2>();
}

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<const T, 1>)
{
return o << test_print<T, 2>() << " const";
}

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<volatile T, 1>)
{
return o << test_print<T, 2>() << " volatile";
}

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<const volatile T, 1>)
{
return o << test_print<T, 2>() << " const volatile";
}

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<T>)
{
return o << test_print<T, 1>();
}

template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<T&>)
{
return o << test_print<T, 1>() << " &";
}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
template<class T> inline std::ostream& operator<<(std::ostream& o, test_print<T&&>)
{
return o << test_print<T, 1>() << " &&";
}
#endif

template< class T > inline void test_trait_impl( char const * trait, void (*)( T ),
bool expected, char const * file, int line, char const * function )
{
if( T::value == expected )
{
test_results();
}
else
{
BOOST_LIGHTWEIGHT_TEST_OSTREAM
<< file << "(" << line << "): predicate '" << trait << "' ["
<< boost::core::demangled_name( BOOST_CORE_TYPEID(T) ) << "]"
<< " test failed in function '" << function
<< "' (should have been " << ( expected? "true": "false" ) << ")"
<< std::endl;

++test_results().errors();
}
}

template<class T> inline bool test_trait_same_impl_( T )
{
return T::value;
}

template<class T1, class T2> inline void test_trait_same_impl( char const * types,
boost::core::is_same<T1, T2> same, char const * file, int line, char const * function )
{
if( test_trait_same_impl_( same ) )
{
test_results();
}
else
{
BOOST_LIGHTWEIGHT_TEST_OSTREAM
<< file << "(" << line << "): test 'is_same<" << types << ">'"
<< " failed in function '" << function
<< "' ('" << test_print<T1>()
<< "' != '" << test_print<T2>() << "')"
<< std::endl;

++test_results().errors();
}
}

} 

} 

#define BOOST_TEST_TRAIT_TRUE(type) ( ::boost::detail::test_trait_impl(#type, (void(*)type)0, true, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION) )
#define BOOST_TEST_TRAIT_FALSE(type) ( ::boost::detail::test_trait_impl(#type, (void(*)type)0, false, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION) )

#if defined(__GNUC__)
# pragma GCC system_header
#endif

#define BOOST_TEST_TRAIT_SAME(...) ( ::boost::detail::test_trait_same_impl(#__VA_ARGS__, ::boost::core::is_same<__VA_ARGS__>(), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION) )

#endif 
