
#if !defined(BOOST_SPIRIT_UNUSED_APRIL_16_2006_0616PM)
#define BOOST_SPIRIT_UNUSED_APRIL_16_2006_0616PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/mpl/bool.hpp>

namespace boost { namespace spirit
{
struct unused_type
{
BOOST_DEFAULTED_FUNCTION(unused_type(), {})

template <typename T>
unused_type(T const&)
{
}

template <typename T>
unused_type const&
operator=(T const&) const
{
return *this;
}

template <typename T>
unused_type&
operator=(T const&)
{
return *this;
}
};

unused_type const unused = unused_type();

namespace detail
{
struct unused_only
{
unused_only(unused_type const&) {}
};
}

template <typename Out>
inline Out& operator<<(Out& out, detail::unused_only const&)
{
return out;
}

template <typename In>
inline In& operator>>(In& in, unused_type&)
{
return in;
}

namespace traits
{
template <typename T> struct not_is_unused : mpl::true_ {};
template <> struct not_is_unused<unused_type> : mpl::false_ {};
}
}}

#endif
