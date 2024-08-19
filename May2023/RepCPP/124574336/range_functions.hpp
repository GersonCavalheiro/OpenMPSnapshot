
#if !defined(BOOST_SPIRIT_RANGE_FUNCTIONS_MAY_16_2006_0720_PM)
#define BOOST_SPIRIT_RANGE_FUNCTIONS_MAY_16_2006_0720_PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/integer_traits.hpp>

namespace boost { namespace spirit { namespace support { namespace detail
{
template <typename Range>
inline bool
is_valid(Range const& range)
{
return range.first <= range.last;
}

template <typename Range>
inline bool
includes(Range const& range, Range const& other)
{
return (range.first <= other.first) && (range.last >= other.last);
}

template <typename Range>
inline bool
includes(Range const& range, typename Range::value_type val)
{
return (range.first <= val) && (range.last >= val);
}

template <typename Range>
inline bool
can_merge(Range const& range, Range const& other)
{

typedef typename Range::value_type value_type;
typedef integer_traits<value_type> integer_traits;

value_type decr_first =
range.first == integer_traits::const_min
? range.first : range.first-1;

value_type incr_last =
range.last == integer_traits::const_max
? range.last : range.last+1;

return (decr_first <= other.last) && (incr_last >= other.first);
}

template <typename Range>
inline void
merge(Range& result, Range const& other)
{
if (result.first > other.first)
result.first = other.first;
if (result.last < other.last)
result.last = other.last;
}

template <typename Range>
struct range_compare
{

typedef typename Range::value_type value_type;

bool operator()(Range const& x, const value_type y) const
{
return x.first < y;
}

bool operator()(value_type const x, Range const& y) const
{
return x < y.first;
}

bool operator()(Range const& x, Range const& y) const
{
return x.first < y.first;
}
};
}}}}

#endif
