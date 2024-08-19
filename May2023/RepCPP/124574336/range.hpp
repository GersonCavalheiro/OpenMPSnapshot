
#if !defined(BOOST_SPIRIT_RANGE_MAY_16_2006_0720_PM)
#define BOOST_SPIRIT_RANGE_MAY_16_2006_0720_PM

#if defined(_MSC_VER)
#pragma once
#endif

namespace boost { namespace spirit { namespace support { namespace detail
{
template <typename T>
struct range
{
typedef T value_type;

range() : first(), last() {}
range(T first_, T last_) : first(first_), last(last_) {}

T first;
T last;
};
}}}}

#endif
