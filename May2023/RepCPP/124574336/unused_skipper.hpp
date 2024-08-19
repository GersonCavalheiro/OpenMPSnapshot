
#if !defined(BOOST_SPIRIT_QI_UNUSED_SKIPPER_JUL_25_2009_0921AM)
#define BOOST_SPIRIT_QI_UNUSED_SKIPPER_JUL_25_2009_0921AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/mpl/bool.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Skipper>
struct unused_skipper : unused_type
{
unused_skipper(Skipper const& skipper_)
: skipper(skipper_) {}
Skipper const& skipper;

BOOST_DELETED_FUNCTION(unused_skipper& operator= (unused_skipper const&))
};

template <typename Skipper>
struct is_unused_skipper
: mpl::false_ {};

template <typename Skipper>
struct is_unused_skipper<unused_skipper<Skipper> >
: mpl::true_ {};

template <>
struct is_unused_skipper<unused_type>
: mpl::true_ {};

template <typename Skipper>
inline Skipper const&
get_skipper(unused_skipper<Skipper> const& u)
{
return u.skipper;
}

template <typename Skipper>
inline Skipper const&
get_skipper(Skipper const& u)
{
return u;
}

}}}}

#endif
