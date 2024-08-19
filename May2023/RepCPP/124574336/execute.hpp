

#ifndef BOOST_IOSTREAMS_DETAIL_EXECUTE_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_EXECUTE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/config/limits.hpp>   
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/utility/result_of.hpp>

namespace boost { namespace iostreams { namespace detail {

template<typename Result>
struct execute_traits_impl {
typedef Result result_type;
template<typename Op>
static Result execute(Op op) { return op(); }
};

template<>
struct execute_traits_impl<void> {
typedef int result_type;
template<typename Op>
static int execute(Op op) { op(); return 0; }
};

template< typename Op, 
typename Result = 
#if !defined(BOOST_NO_RESULT_OF) && \
!BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x592))
typename boost::result_of<Op()>::type
#else
BOOST_DEDUCED_TYPENAME Op::result_type
#endif
>
struct execute_traits 
: execute_traits_impl<Result>
{ };

template<typename Op>
typename execute_traits<Op>::result_type 
execute_all(Op op) 
{ 
return execute_traits<Op>::execute(op);
}

#define BOOST_PP_LOCAL_MACRO(n) \
template<typename Op, BOOST_PP_ENUM_PARAMS(n, typename C)> \
typename execute_traits<Op>::result_type \
execute_all(Op op, BOOST_PP_ENUM_BINARY_PARAMS(n, C, c)) \
{ \
typename execute_traits<Op>::result_type r; \
try { \
r = boost::iostreams::detail::execute_all( \
op BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(n), c) \
); \
} catch (...) { \
try { \
BOOST_PP_CAT(c, BOOST_PP_DEC(n))(); \
} catch (...) { } \
throw; \
} \
BOOST_PP_CAT(c, BOOST_PP_DEC(n))(); \
return r; \
} \


#define BOOST_PP_LOCAL_LIMITS (1, BOOST_IOSTREAMS_MAX_EXECUTE_ARITY)
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_LOCAL_MACRO

template<class InIt, class Op>
Op execute_foreach(InIt first, InIt last, Op op)
{
if (first == last)
return op;
try {
op(*first);
} catch (...) {
try {
++first;
boost::iostreams::detail::execute_foreach(first, last, op);
} catch (...) { }
throw;
}
++first;
return boost::iostreams::detail::execute_foreach(first, last, op);
}

} } } 

#endif 
