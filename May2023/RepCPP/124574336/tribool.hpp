


#ifndef BOOST_LOGIC_TRIBOOL_HPP
#define BOOST_LOGIC_TRIBOOL_HPP

#include <boost/logic/tribool_fwd.hpp>
#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#  pragma once
#endif

namespace boost { namespace logic {

namespace detail {

struct indeterminate_t
{
#if BOOST_WORKAROUND(BOOST_BORLANDC, < 0x0600)
char dummy_; 
#endif
};

} 


typedef bool (*indeterminate_keyword_t)(tribool, detail::indeterminate_t);


BOOST_CONSTEXPR inline bool
indeterminate(tribool x,
detail::indeterminate_t dummy = detail::indeterminate_t()) BOOST_NOEXCEPT;


class tribool
{
#if defined( BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS )
private:
struct dummy {
void nonnull() {};
};

typedef void (dummy::*safe_bool)();
#endif

public:

BOOST_CONSTEXPR tribool() BOOST_NOEXCEPT : value(false_value) {}


BOOST_CONSTEXPR tribool(bool initial_value) BOOST_NOEXCEPT : value(initial_value? true_value : false_value) {}


BOOST_CONSTEXPR tribool(indeterminate_keyword_t) BOOST_NOEXCEPT : value(indeterminate_value) {}


#if !defined( BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS )

BOOST_CONSTEXPR explicit operator bool () const BOOST_NOEXCEPT
{
return value == true_value;
}

#else

BOOST_CONSTEXPR operator safe_bool() const BOOST_NOEXCEPT
{
return value == true_value? &dummy::nonnull : 0;
}

#endif


enum value_t { false_value, true_value, indeterminate_value } value;
};

BOOST_CONSTEXPR inline bool indeterminate(tribool x, detail::indeterminate_t) BOOST_NOEXCEPT
{
return x.value == tribool::indeterminate_value;
}



BOOST_CONSTEXPR inline tribool operator!(tribool x) BOOST_NOEXCEPT
{
return x.value == tribool::false_value? tribool(true)
:x.value == tribool::true_value? tribool(false)
:tribool(indeterminate);
}


BOOST_CONSTEXPR inline tribool operator&&(tribool x, tribool y) BOOST_NOEXCEPT
{
return (static_cast<bool>(!x) || static_cast<bool>(!y))
? tribool(false)
: ((static_cast<bool>(x) && static_cast<bool>(y)) ? tribool(true) : indeterminate)
;
}


BOOST_CONSTEXPR inline tribool operator&&(tribool x, bool y) BOOST_NOEXCEPT
{ return y? x : tribool(false); }


BOOST_CONSTEXPR inline tribool operator&&(bool x, tribool y) BOOST_NOEXCEPT
{ return x? y : tribool(false); }


BOOST_CONSTEXPR inline tribool operator&&(indeterminate_keyword_t, tribool x) BOOST_NOEXCEPT
{ return !x? tribool(false) : tribool(indeterminate); }


BOOST_CONSTEXPR inline tribool operator&&(tribool x, indeterminate_keyword_t) BOOST_NOEXCEPT
{ return !x? tribool(false) : tribool(indeterminate); }


BOOST_CONSTEXPR inline tribool operator||(tribool x, tribool y) BOOST_NOEXCEPT
{
return (static_cast<bool>(!x) && static_cast<bool>(!y))
? tribool(false)
: ((static_cast<bool>(x) || static_cast<bool>(y)) ? tribool(true) : tribool(indeterminate))
;
}


BOOST_CONSTEXPR inline tribool operator||(tribool x, bool y) BOOST_NOEXCEPT
{ return y? tribool(true) : x; }


BOOST_CONSTEXPR inline tribool operator||(bool x, tribool y) BOOST_NOEXCEPT
{ return x? tribool(true) : y; }


BOOST_CONSTEXPR inline tribool operator||(indeterminate_keyword_t, tribool x) BOOST_NOEXCEPT
{ return x? tribool(true) : tribool(indeterminate); }


BOOST_CONSTEXPR inline tribool operator||(tribool x, indeterminate_keyword_t) BOOST_NOEXCEPT
{ return x? tribool(true) : tribool(indeterminate); }


BOOST_CONSTEXPR inline tribool operator==(tribool x, tribool y) BOOST_NOEXCEPT
{
return (indeterminate(x) || indeterminate(y))
? indeterminate
: ((x && y) || (!x && !y))
;
}


BOOST_CONSTEXPR inline tribool operator==(tribool x, bool y) BOOST_NOEXCEPT { return x == tribool(y); }


BOOST_CONSTEXPR inline tribool operator==(bool x, tribool y) BOOST_NOEXCEPT { return tribool(x) == y; }


BOOST_CONSTEXPR inline tribool operator==(indeterminate_keyword_t, tribool x) BOOST_NOEXCEPT
{ return tribool(indeterminate) == x; }


BOOST_CONSTEXPR inline tribool operator==(tribool x, indeterminate_keyword_t) BOOST_NOEXCEPT
{ return tribool(indeterminate) == x; }


BOOST_CONSTEXPR inline tribool operator!=(tribool x, tribool y) BOOST_NOEXCEPT
{
return (indeterminate(x) || indeterminate(y))
? indeterminate
: !((x && y) || (!x && !y))
;
}


BOOST_CONSTEXPR inline tribool operator!=(tribool x, bool y) BOOST_NOEXCEPT { return x != tribool(y); }


BOOST_CONSTEXPR inline tribool operator!=(bool x, tribool y) BOOST_NOEXCEPT { return tribool(x) != y; }


BOOST_CONSTEXPR inline tribool operator!=(indeterminate_keyword_t, tribool x) BOOST_NOEXCEPT
{ return tribool(indeterminate) != x; }


BOOST_CONSTEXPR inline tribool operator!=(tribool x, indeterminate_keyword_t) BOOST_NOEXCEPT
{ return x != tribool(indeterminate); }

} } 

namespace boost {
using logic::tribool;
using logic::indeterminate;
}


#define BOOST_TRIBOOL_THIRD_STATE(Name)                                 \
inline bool                                                             \
Name(boost::logic::tribool x,                                           \
boost::logic::detail::indeterminate_t =                            \
boost::logic::detail::indeterminate_t())                         \
{ return x.value == boost::logic::tribool::indeterminate_value; }

#endif 

