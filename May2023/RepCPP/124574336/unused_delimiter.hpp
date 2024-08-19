
#if !defined(BOOST_SPIRIT_KARMA_UNUSED_DELIMITER_MAR_15_2009_0923PM)
#define BOOST_SPIRIT_KARMA_UNUSED_DELIMITER_MAR_15_2009_0923PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>

namespace boost { namespace spirit { namespace karma { namespace detail
{
template <typename Delimiter>
struct unused_delimiter : unused_type
{
unused_delimiter(Delimiter const& delim)
: delimiter(delim) {}
Delimiter const& delimiter;

BOOST_DELETED_FUNCTION(unused_delimiter& operator= (unused_delimiter const&))
};

template <typename Delimiter, typename Default>
inline Delimiter const& 
get_delimiter(unused_delimiter<Delimiter> const& u, Default const&)
{
return u.delimiter;
}

template <typename Delimiter, typename Default>
inline Default const& 
get_delimiter(Delimiter const&, Default const& d)
{
return d;
}

}}}}

#endif
