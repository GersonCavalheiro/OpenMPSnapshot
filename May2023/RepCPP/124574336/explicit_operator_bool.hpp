



#ifndef BOOST_CORE_EXPLICIT_OPERATOR_BOOL_HPP
#define BOOST_CORE_EXPLICIT_OPERATOR_BOOL_HPP

#include <boost/config.hpp>
#include <boost/config/workaround.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined(BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS)


#define BOOST_EXPLICIT_OPERATOR_BOOL()\
BOOST_FORCEINLINE explicit operator bool () const\
{\
return !this->operator! ();\
}


#define BOOST_EXPLICIT_OPERATOR_BOOL_NOEXCEPT()\
BOOST_FORCEINLINE explicit operator bool () const BOOST_NOEXCEPT\
{\
return !this->operator! ();\
}

#if !BOOST_WORKAROUND(BOOST_GCC, < 40700)


#define BOOST_CONSTEXPR_EXPLICIT_OPERATOR_BOOL()\
BOOST_FORCEINLINE BOOST_CONSTEXPR explicit operator bool () const BOOST_NOEXCEPT\
{\
return !this->operator! ();\
}

#else

#define BOOST_CONSTEXPR_EXPLICIT_OPERATOR_BOOL() BOOST_EXPLICIT_OPERATOR_BOOL_NOEXCEPT()

#endif

#else 

#if (defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x530)) && !defined(BOOST_NO_COMPILER_CONFIG)
#define BOOST_NO_UNSPECIFIED_BOOL
#endif 

#if !defined(BOOST_NO_UNSPECIFIED_BOOL)

namespace boost {

namespace detail {

#if !defined(_MSC_VER) && !defined(__IBMCPP__)

struct unspecified_bool
{
struct OPERATORS_NOT_ALLOWED;
static void true_value(OPERATORS_NOT_ALLOWED*) {}
};
typedef void (*unspecified_bool_type)(unspecified_bool::OPERATORS_NOT_ALLOWED*);

#else

struct unspecified_bool
{
struct OPERATORS_NOT_ALLOWED;
void true_value(OPERATORS_NOT_ALLOWED*) {}
};
typedef void (unspecified_bool::*unspecified_bool_type)(unspecified_bool::OPERATORS_NOT_ALLOWED*);

#endif

} 

} 

#define BOOST_EXPLICIT_OPERATOR_BOOL()\
BOOST_FORCEINLINE operator boost::detail::unspecified_bool_type () const\
{\
return (!this->operator! () ? &boost::detail::unspecified_bool::true_value : (boost::detail::unspecified_bool_type)0);\
}

#define BOOST_EXPLICIT_OPERATOR_BOOL_NOEXCEPT()\
BOOST_FORCEINLINE operator boost::detail::unspecified_bool_type () const BOOST_NOEXCEPT\
{\
return (!this->operator! () ? &boost::detail::unspecified_bool::true_value : (boost::detail::unspecified_bool_type)0);\
}

#define BOOST_CONSTEXPR_EXPLICIT_OPERATOR_BOOL()\
BOOST_FORCEINLINE BOOST_CONSTEXPR operator boost::detail::unspecified_bool_type () const BOOST_NOEXCEPT\
{\
return (!this->operator! () ? &boost::detail::unspecified_bool::true_value : (boost::detail::unspecified_bool_type)0);\
}

#else 

#define BOOST_EXPLICIT_OPERATOR_BOOL()\
BOOST_FORCEINLINE operator bool () const\
{\
return !this->operator! ();\
}

#define BOOST_EXPLICIT_OPERATOR_BOOL_NOEXCEPT()\
BOOST_FORCEINLINE operator bool () const BOOST_NOEXCEPT\
{\
return !this->operator! ();\
}

#define BOOST_CONSTEXPR_EXPLICIT_OPERATOR_BOOL()\
BOOST_FORCEINLINE BOOST_CONSTEXPR operator bool () const BOOST_NOEXCEPT\
{\
return !this->operator! ();\
}

#endif 

#endif 

#endif 
