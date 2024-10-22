
#ifndef BOOST_XPRESSIVE_DETAIL_UTILITY_WIDTH_HPP_EAN_04_07_2006
#define BOOST_XPRESSIVE_DETAIL_UTILITY_WIDTH_HPP_EAN_04_07_2006

#if defined(_MSC_VER)
# pragma once
#endif

#include <climits> 
#include <boost/mpl/size_t.hpp>

namespace boost { namespace xpressive { namespace detail
{

typedef mpl::size_t<INT_MAX / 2 - 1> unknown_width;
struct width;
bool is_unknown(width const &that);

struct width
{
width(std::size_t val = 0)
: value_(val)
{
}

bool operator !() const
{
return !this->value_;
}

width &operator +=(width const &that)
{
this->value_ =
!is_unknown(*this) && !is_unknown(that)
? this->value_ + that.value_
: unknown_width();
return *this;
}

width &operator |=(width const &that)
{
this->value_ =
this->value_ == that.value_
? this->value_
: unknown_width();
return *this;
}

std::size_t value() const
{
return this->value_;
}

private:
std::size_t value_;
};

inline bool is_unknown(width const &that)
{
return unknown_width::value == that.value();
}

inline bool operator ==(width const &left, width const &right)
{
return left.value() == right.value();
}

inline bool operator !=(width const &left, width const &right)
{
return left.value() != right.value();
}

inline width operator +(width left, width const &right)
{
return left += right;
}

inline width operator |(width left, width const &right)
{
return left |= right;
}

}}} 

#endif
