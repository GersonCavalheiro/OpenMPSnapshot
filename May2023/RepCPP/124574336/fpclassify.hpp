
#ifndef BOOST_MATH_FPCLASSIFY_HPP
#define BOOST_MATH_FPCLASSIFY_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <math.h>
#include <boost/config/no_tr1/cmath.hpp>
#include <boost/limits.hpp>
#include <boost/math/tools/real_cast.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/detail/fp_traits.hpp>




#if defined(_MSC_VER) || defined(BOOST_BORLANDC)
#include <float.h>
#endif
#ifdef BOOST_MATH_USE_FLOAT128
#ifdef __has_include
#if  __has_include("quadmath.h")
#include "quadmath.h"
#define BOOST_MATH_HAS_QUADMATH_H
#endif
#endif
#endif

#ifdef BOOST_NO_STDC_NAMESPACE
namespace std{ using ::abs; using ::fabs; }
#endif

namespace boost{

namespace math_detail{

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4800)
#endif

template <class T>
inline bool is_nan_helper(T t, const boost::true_type&)
{
#ifdef isnan
return isnan(t);
#elif defined(BOOST_MATH_DISABLE_STD_FPCLASSIFY) || !defined(BOOST_HAS_FPCLASSIFY)
(void)t;
return false;
#else 
return (BOOST_FPCLASSIFY_PREFIX fpclassify(t) == (int)FP_NAN);
#endif
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class T>
inline bool is_nan_helper(T, const boost::false_type&)
{
return false;
}
#if defined(BOOST_MATH_USE_FLOAT128) 
#if defined(BOOST_MATH_HAS_QUADMATH_H)
inline bool is_nan_helper(__float128 f, const boost::true_type&) { return ::isnanq(f); }
inline bool is_nan_helper(__float128 f, const boost::false_type&) { return ::isnanq(f); }
#elif defined(BOOST_GNU_STDLIB) && BOOST_GNU_STDLIB && \
_GLIBCXX_USE_C99_MATH && !_GLIBCXX_USE_C99_FP_MACROS_DYNAMIC
inline bool is_nan_helper(__float128 f, const boost::true_type&) { return std::isnan(static_cast<double>(f)); }
inline bool is_nan_helper(__float128 f, const boost::false_type&) { return std::isnan(static_cast<double>(f)); }
#else
inline bool is_nan_helper(__float128 f, const boost::true_type&) { return ::isnan(static_cast<double>(f)); }
inline bool is_nan_helper(__float128 f, const boost::false_type&) { return ::isnan(static_cast<double>(f)); }
#endif
#endif
}

namespace math{

namespace detail{

#ifdef BOOST_MATH_USE_STD_FPCLASSIFY
template <class T>
inline int fpclassify_imp BOOST_NO_MACRO_EXPAND(T t, const native_tag&)
{
return (std::fpclassify)(t);
}
#endif

template <class T>
inline int fpclassify_imp BOOST_NO_MACRO_EXPAND(T t, const generic_tag<true>&)
{
BOOST_MATH_INSTRUMENT_VARIABLE(t);

#if defined(BOOST_HAS_FPCLASSIFY)  && !defined(BOOST_MATH_DISABLE_STD_FPCLASSIFY)
if(::boost::math_detail::is_nan_helper(t, ::boost::is_floating_point<T>()))
return FP_NAN;
#elif defined(isnan)
if(boost::math_detail::is_nan_helper(t, ::boost::is_floating_point<T>()))
return FP_NAN;
#elif defined(_MSC_VER) || defined(BOOST_BORLANDC)
if(::_isnan(boost::math::tools::real_cast<double>(t)))
return FP_NAN;
#endif
T at = (t < T(0)) ? -t : t;

if(at <= (std::numeric_limits<T>::max)())
{
if(at >= (std::numeric_limits<T>::min)())
return FP_NORMAL;
return (at != 0) ? FP_SUBNORMAL : FP_ZERO;
}
else if(at > (std::numeric_limits<T>::max)())
return FP_INFINITE;
return FP_NAN;
}

template <class T>
inline int fpclassify_imp BOOST_NO_MACRO_EXPAND(T t, const generic_tag<false>&)
{
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<T>::is_specialized)
return fpclassify_imp(t, generic_tag<true>());
#endif
BOOST_MATH_INSTRUMENT_VARIABLE(t);

return t == 0 ? FP_ZERO : FP_NORMAL;
}

template<class T>
int fpclassify_imp BOOST_NO_MACRO_EXPAND(T x, ieee_copy_all_bits_tag)
{
typedef BOOST_DEDUCED_TYPENAME fp_traits<T>::type traits;

BOOST_MATH_INSTRUMENT_VARIABLE(x);

BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
BOOST_MATH_INSTRUMENT_VARIABLE(a);
a &= traits::exponent | traits::flag | traits::significand;
BOOST_MATH_INSTRUMENT_VARIABLE((traits::exponent | traits::flag | traits::significand));
BOOST_MATH_INSTRUMENT_VARIABLE(a);

if(a <= traits::significand) {
if(a == 0)
return FP_ZERO;
else
return FP_SUBNORMAL;
}

if(a < traits::exponent) return FP_NORMAL;

a &= traits::significand;
if(a == 0) return FP_INFINITE;

return FP_NAN;
}

template<class T>
int fpclassify_imp BOOST_NO_MACRO_EXPAND(T x, ieee_copy_leading_bits_tag)
{
typedef BOOST_DEDUCED_TYPENAME fp_traits<T>::type traits;

BOOST_MATH_INSTRUMENT_VARIABLE(x);

BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
a &= traits::exponent | traits::flag | traits::significand;

if(a <= traits::significand) {
if(x == 0)
return FP_ZERO;
else
return FP_SUBNORMAL;
}

if(a < traits::exponent) return FP_NORMAL;

a &= traits::significand;
traits::set_bits(x,a);
if(x == 0) return FP_INFINITE;

return FP_NAN;
}

#if defined(BOOST_MATH_USE_STD_FPCLASSIFY) && (defined(BOOST_MATH_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY) || defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS))
inline int fpclassify_imp BOOST_NO_MACRO_EXPAND(long double t, const native_tag&)
{
return boost::math::detail::fpclassify_imp(t, generic_tag<true>());
}
#endif

}  

template <class T>
inline int fpclassify BOOST_NO_MACRO_EXPAND(T t)
{
typedef typename detail::fp_traits<T>::type traits;
typedef typename traits::method method;
typedef typename tools::promote_args_permissive<T>::type value_type;
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<T>::is_specialized && detail::is_generic_tag_false(static_cast<method*>(0)))
return detail::fpclassify_imp(static_cast<value_type>(t), detail::generic_tag<true>());
return detail::fpclassify_imp(static_cast<value_type>(t), method());
#else
return detail::fpclassify_imp(static_cast<value_type>(t), method());
#endif
}

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
template <>
inline int fpclassify<long double> BOOST_NO_MACRO_EXPAND(long double t)
{
typedef detail::fp_traits<long double>::type traits;
typedef traits::method method;
typedef long double value_type;
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<long double>::is_specialized && detail::is_generic_tag_false(static_cast<method*>(0)))
return detail::fpclassify_imp(static_cast<value_type>(t), detail::generic_tag<true>());
return detail::fpclassify_imp(static_cast<value_type>(t), method());
#else
return detail::fpclassify_imp(static_cast<value_type>(t), method());
#endif
}
#endif

namespace detail {

#ifdef BOOST_MATH_USE_STD_FPCLASSIFY
template<class T>
inline bool isfinite_impl(T x, native_tag const&)
{
return (std::isfinite)(x);
}
#endif

template<class T>
inline bool isfinite_impl(T x, generic_tag<true> const&)
{
return x >= -(std::numeric_limits<T>::max)()
&& x <= (std::numeric_limits<T>::max)();
}

template<class T>
inline bool isfinite_impl(T x, generic_tag<false> const&)
{
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<T>::is_specialized)
return isfinite_impl(x, generic_tag<true>());
#endif
(void)x; 
return true;
}

template<class T>
inline bool isfinite_impl(T x, ieee_tag const&)
{
typedef BOOST_DEDUCED_TYPENAME detail::fp_traits<T>::type traits;
BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
a &= traits::exponent;
return a != traits::exponent;
}

#if defined(BOOST_MATH_USE_STD_FPCLASSIFY) && defined(BOOST_MATH_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY)
inline bool isfinite_impl BOOST_NO_MACRO_EXPAND(long double t, const native_tag&)
{
return boost::math::detail::isfinite_impl(t, generic_tag<true>());
}
#endif

}

template<class T>
inline bool (isfinite)(T x)
{ 
typedef typename detail::fp_traits<T>::type traits;
typedef typename traits::method method;
typedef typename tools::promote_args_permissive<T>::type value_type;
return detail::isfinite_impl(static_cast<value_type>(x), method());
}

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
template<>
inline bool (isfinite)(long double x)
{ 
typedef detail::fp_traits<long double>::type traits;
typedef traits::method method;
typedef long double value_type;
return detail::isfinite_impl(static_cast<value_type>(x), method());
}
#endif


namespace detail {

#ifdef BOOST_MATH_USE_STD_FPCLASSIFY
template<class T>
inline bool isnormal_impl(T x, native_tag const&)
{
return (std::isnormal)(x);
}
#endif

template<class T>
inline bool isnormal_impl(T x, generic_tag<true> const&)
{
if(x < 0) x = -x;
return x >= (std::numeric_limits<T>::min)()
&& x <= (std::numeric_limits<T>::max)();
}

template<class T>
inline bool isnormal_impl(T x, generic_tag<false> const&)
{
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<T>::is_specialized)
return isnormal_impl(x, generic_tag<true>());
#endif
return !(x == 0);
}

template<class T>
inline bool isnormal_impl(T x, ieee_tag const&)
{
typedef BOOST_DEDUCED_TYPENAME detail::fp_traits<T>::type traits;
BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
a &= traits::exponent | traits::flag;
return (a != 0) && (a < traits::exponent);
}

#if defined(BOOST_MATH_USE_STD_FPCLASSIFY) && defined(BOOST_MATH_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY)
inline bool isnormal_impl BOOST_NO_MACRO_EXPAND(long double t, const native_tag&)
{
return boost::math::detail::isnormal_impl(t, generic_tag<true>());
}
#endif

}

template<class T>
inline bool (isnormal)(T x)
{
typedef typename detail::fp_traits<T>::type traits;
typedef typename traits::method method;
typedef typename tools::promote_args_permissive<T>::type value_type;
return detail::isnormal_impl(static_cast<value_type>(x), method());
}

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
template<>
inline bool (isnormal)(long double x)
{
typedef detail::fp_traits<long double>::type traits;
typedef traits::method method;
typedef long double value_type;
return detail::isnormal_impl(static_cast<value_type>(x), method());
}
#endif


namespace detail {

#ifdef BOOST_MATH_USE_STD_FPCLASSIFY
template<class T>
inline bool isinf_impl(T x, native_tag const&)
{
return (std::isinf)(x);
}
#endif

template<class T>
inline bool isinf_impl(T x, generic_tag<true> const&)
{
(void)x; 
return std::numeric_limits<T>::has_infinity
&& ( x == std::numeric_limits<T>::infinity()
|| x == -std::numeric_limits<T>::infinity());
}

template<class T>
inline bool isinf_impl(T x, generic_tag<false> const&)
{
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<T>::is_specialized)
return isinf_impl(x, generic_tag<true>());
#endif
(void)x; 
return false;
}

template<class T>
inline bool isinf_impl(T x, ieee_copy_all_bits_tag const&)
{
typedef BOOST_DEDUCED_TYPENAME fp_traits<T>::type traits;

BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
a &= traits::exponent | traits::significand;
return a == traits::exponent;
}

template<class T>
inline bool isinf_impl(T x, ieee_copy_leading_bits_tag const&)
{
typedef BOOST_DEDUCED_TYPENAME fp_traits<T>::type traits;

BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
a &= traits::exponent | traits::significand;
if(a != traits::exponent)
return false;

traits::set_bits(x,0);
return x == 0;
}

#if defined(BOOST_MATH_USE_STD_FPCLASSIFY) && defined(BOOST_MATH_NO_NATIVE_LONG_DOUBLE_FP_CLASSIFY)
inline bool isinf_impl BOOST_NO_MACRO_EXPAND(long double t, const native_tag&)
{
return boost::math::detail::isinf_impl(t, generic_tag<true>());
}
#endif

}   

template<class T>
inline bool (isinf)(T x)
{
typedef typename detail::fp_traits<T>::type traits;
typedef typename traits::method method;
typedef typename tools::promote_args_permissive<T>::type value_type;
return detail::isinf_impl(static_cast<value_type>(x), method());
}

#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
template<>
inline bool (isinf)(long double x)
{
typedef detail::fp_traits<long double>::type traits;
typedef traits::method method;
typedef long double value_type;
return detail::isinf_impl(static_cast<value_type>(x), method());
}
#endif
#if defined(BOOST_MATH_USE_FLOAT128) && defined(BOOST_MATH_HAS_QUADMATH_H)
template<>
inline bool (isinf)(__float128 x)
{
return ::isinfq(x);
}
#endif


namespace detail {

#ifdef BOOST_MATH_USE_STD_FPCLASSIFY
template<class T>
inline bool isnan_impl(T x, native_tag const&)
{
return (std::isnan)(x);
}
#endif

template<class T>
inline bool isnan_impl(T x, generic_tag<true> const&)
{
return std::numeric_limits<T>::has_infinity
? !(x <= std::numeric_limits<T>::infinity())
: x != x;
}

template<class T>
inline bool isnan_impl(T x, generic_tag<false> const&)
{
#ifdef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
if(std::numeric_limits<T>::is_specialized)
return isnan_impl(x, generic_tag<true>());
#endif
(void)x; 
return false;
}

template<class T>
inline bool isnan_impl(T x, ieee_copy_all_bits_tag const&)
{
typedef BOOST_DEDUCED_TYPENAME fp_traits<T>::type traits;

BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);
a &= traits::exponent | traits::significand;
return a > traits::exponent;
}

template<class T>
inline bool isnan_impl(T x, ieee_copy_leading_bits_tag const&)
{
typedef BOOST_DEDUCED_TYPENAME fp_traits<T>::type traits;

BOOST_DEDUCED_TYPENAME traits::bits a;
traits::get_bits(x,a);

a &= traits::exponent | traits::significand;
if(a < traits::exponent)
return false;

a &= traits::significand;
traits::set_bits(x,a);
return x != 0;
}

}   

template<class T>
inline bool (isnan)(T x)
{ 
typedef typename detail::fp_traits<T>::type traits;
typedef typename traits::method method;
return detail::isnan_impl(x, method());
}

#ifdef isnan
template <> inline bool isnan BOOST_NO_MACRO_EXPAND<float>(float t){ return ::boost::math_detail::is_nan_helper(t, boost::true_type()); }
template <> inline bool isnan BOOST_NO_MACRO_EXPAND<double>(double t){ return ::boost::math_detail::is_nan_helper(t, boost::true_type()); }
template <> inline bool isnan BOOST_NO_MACRO_EXPAND<long double>(long double t){ return ::boost::math_detail::is_nan_helper(t, boost::true_type()); }
#elif defined(BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS)
template<>
inline bool (isnan)(long double x)
{ 
typedef detail::fp_traits<long double>::type traits;
typedef traits::method method;
return detail::isnan_impl(x, method());
}
#endif
#if defined(BOOST_MATH_USE_FLOAT128) && defined(BOOST_MATH_HAS_QUADMATH_H)
template<>
inline bool (isnan)(__float128 x)
{
return ::isnanq(x);
}
#endif

} 
} 

#endif 

