



#ifndef BOOST_OLD_NUMERIC_CAST_HPP
#define BOOST_OLD_NUMERIC_CAST_HPP

# include <boost/config.hpp>
# include <cassert>
# include <typeinfo>
# include <boost/type.hpp>
# include <boost/limits.hpp>
# include <boost/numeric/conversion/converter_policies.hpp>

namespace boost
{
using numeric::bad_numeric_cast;




#if !defined(BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS) || defined(BOOST_SGI_CPP_LIMITS)

namespace detail
{
template <class T>
struct signed_numeric_limits : std::numeric_limits<T>
{
static inline T min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return (std::numeric_limits<T>::min)() >= 0
? static_cast<T>(-(std::numeric_limits<T>::max)())
: (std::numeric_limits<T>::min)();
};
};

template <class T, bool specialized>
struct fixed_numeric_limits_base
: public if_true< std::numeric_limits<T>::is_signed >
::BOOST_NESTED_TEMPLATE then< signed_numeric_limits<T>,
std::numeric_limits<T>
>::type
{};

template <class T>
struct fixed_numeric_limits
: fixed_numeric_limits_base<T,(std::numeric_limits<T>::is_specialized)>
{};

# ifdef BOOST_HAS_LONG_LONG
template <>
struct fixed_numeric_limits_base< ::boost::long_long_type, false>
{
BOOST_STATIC_CONSTANT(bool, is_specialized = true);
BOOST_STATIC_CONSTANT(bool, is_signed = true);
static  ::boost::long_long_type max BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
#  ifdef LONGLONG_MAX
return LONGLONG_MAX;
#  else
return 9223372036854775807LL; 
#  endif
}

static  ::boost::long_long_type min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
#  ifdef LONGLONG_MIN
return LONGLONG_MIN;
#  else
return -( 9223372036854775807LL )-1; 
#  endif
}
};

template <>
struct fixed_numeric_limits_base< ::boost::ulong_long_type, false>
{
BOOST_STATIC_CONSTANT(bool, is_specialized = true);
BOOST_STATIC_CONSTANT(bool, is_signed = false);
static  ::boost::ulong_long_type max BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
#  ifdef ULONGLONG_MAX
return ULONGLONG_MAX;
#  else
return 0xffffffffffffffffULL; 
#  endif
}

static  ::boost::ulong_long_type min BOOST_PREVENT_MACRO_SUBSTITUTION () { return 0; }
};
# endif
} 

template <bool x_is_signed, bool y_is_signed>
struct less_than_type_min
{
template <class X, class Y>
static bool check(X x, Y y_min)
{ return x < y_min; }
};

template <>
struct less_than_type_min<false, true>
{
template <class X, class Y>
static bool check(X, Y)
{ return false; }
};

template <>
struct less_than_type_min<true, false>
{
template <class X, class Y>
static bool check(X x, Y)
{ return x < 0; }
};

template <bool same_sign, bool x_is_signed>
struct greater_than_type_max;

template<>
struct greater_than_type_max<true, true>
{
template <class X, class Y>
static inline bool check(X x, Y y_max)
{ return x > y_max; }
};

template <>
struct greater_than_type_max<false, true>
{
template <class X, class Y>
static inline bool check(X x, Y)
{ return x >= 0 && static_cast<X>(static_cast<Y>(x)) != x; }
};

template<>
struct greater_than_type_max<true, false>
{
template <class X, class Y>
static inline bool check(X x, Y y_max)
{ return x > y_max; }
};

template <>
struct greater_than_type_max<false, false>
{
template <class X, class Y>
static inline bool check(X x, Y)
{ return static_cast<X>(static_cast<Y>(x)) != x; }
};

#else 

namespace detail
{
# if BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4018)
#  pragma warning(disable : 4146)
#elif defined(BOOST_BORLANDC)
#  pragma option push -w-8041
# endif

template <class T>
struct fixed_numeric_limits : public std::numeric_limits<T>
{
static inline T min BOOST_PREVENT_MACRO_SUBSTITUTION ()
{
return std::numeric_limits<T>::is_signed && (std::numeric_limits<T>::min)() >= 0
? T(-(std::numeric_limits<T>::max)()) : (std::numeric_limits<T>::min)();
}
};

# if BOOST_MSVC
#  pragma warning(pop)
#elif defined(BOOST_BORLANDC)
#  pragma option pop
# endif
} 

#endif

template<typename Target, typename Source>
inline Target numeric_cast(Source arg)
{
typedef detail::fixed_numeric_limits<Source> arg_traits;
typedef detail::fixed_numeric_limits<Target> result_traits;

#if defined(BOOST_STRICT_CONFIG) \
|| (!defined(__HP_aCC) || __HP_aCC > 33900) \
&& (!defined(BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS) \
|| defined(BOOST_SGI_CPP_LIMITS))
typedef bool argument_must_be_numeric[arg_traits::is_specialized];
typedef bool result_must_be_numeric[result_traits::is_specialized];

const bool arg_is_signed = arg_traits::is_signed;
const bool result_is_signed = result_traits::is_signed;
const bool same_sign = arg_is_signed == result_is_signed;

if (less_than_type_min<arg_is_signed, result_is_signed>::check(arg, (result_traits::min)())
|| greater_than_type_max<same_sign, arg_is_signed>::check(arg, (result_traits::max)())
)

#else 

# if BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4018)
#elif defined(BOOST_BORLANDC)
#pragma option push -w-8012
# endif
if ((arg < 0 && !result_traits::is_signed)  
|| (arg_traits::is_signed && arg < (result_traits::min)())  
|| arg > (result_traits::max)())            
# if BOOST_MSVC
#  pragma warning(pop)
#elif defined(BOOST_BORLANDC)
#pragma option pop
# endif
#endif
{
throw bad_numeric_cast();
}
return static_cast<Target>(arg);
} 

} 

#endif  
