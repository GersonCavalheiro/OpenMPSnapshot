
#ifndef BOOST_XPRESSIVE_SUB_MATCH_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_SUB_MATCH_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <iosfwd>
#include <string>
#include <utility>
#include <iterator>
#include <algorithm>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/range/mutable_iterator.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>

#ifdef BOOST_XPRESSIVE_DOXYGEN_INVOKED
namespace std
{
template<typename, typename> struct pair {};
}
#endif

namespace boost { namespace xpressive
{

template<typename BidiIter>
struct sub_match
: std::pair<BidiIter, BidiIter>
{
private:
struct dummy { int i_; };
typedef int dummy::*bool_type;

public:
typedef typename iterator_value<BidiIter>::type value_type;
typedef typename iterator_difference<BidiIter>::type difference_type;
typedef typename detail::string_type<value_type>::type string_type;
typedef BidiIter iterator;

sub_match()
: std::pair<BidiIter, BidiIter>()
, matched(false)
{
}

sub_match(BidiIter first, BidiIter second, bool matched_ = false)
: std::pair<BidiIter, BidiIter>(first, second)
, matched(matched_)
{
}

string_type str() const
{
return this->matched ? string_type(this->first, this->second) : string_type();
}

operator string_type() const
{
return this->matched ? string_type(this->first, this->second) : string_type();
}

difference_type length() const
{
return this->matched ? std::distance(this->first, this->second) : 0;
}

operator bool_type() const
{
return this->matched ? &dummy::i_ : 0;
}

bool operator !() const
{
return !this->matched;
}

int compare(string_type const &str) const
{
return this->str().compare(str);
}

int compare(sub_match const &sub) const
{
return this->str().compare(sub.str());
}

int compare(value_type const *ptr) const
{
return this->str().compare(ptr);
}

bool matched;
};

template<typename BidiIter>
inline BidiIter range_begin(sub_match<BidiIter> &sub)
{
return sub.first;
}

template<typename BidiIter>
inline BidiIter range_begin(sub_match<BidiIter> const &sub)
{
return sub.first;
}

template<typename BidiIter>
inline BidiIter range_end(sub_match<BidiIter> &sub)
{
return sub.second;
}

template<typename BidiIter>
inline BidiIter range_end(sub_match<BidiIter> const &sub)
{
return sub.second;
}

template<typename BidiIter, typename Char, typename Traits>
inline std::basic_ostream<Char, Traits> &operator <<
(
std::basic_ostream<Char, Traits> &sout
, sub_match<BidiIter> const &sub
)
{
typedef typename iterator_value<BidiIter>::type char_type;
BOOST_MPL_ASSERT_MSG(
(boost::is_same<Char, char_type>::value)
, CHARACTER_TYPES_OF_STREAM_AND_SUB_MATCH_MUST_MATCH
, (Char, char_type)
);
if(sub.matched)
{
std::ostream_iterator<char_type, Char, Traits> iout(sout);
std::copy(sub.first, sub.second, iout);
}
return sout;
}



template<typename BidiIter>
bool operator == (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.compare(rhs) == 0;
}

template<typename BidiIter>
bool operator != (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.compare(rhs) != 0;
}

template<typename BidiIter>
bool operator < (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.compare(rhs) < 0;
}

template<typename BidiIter>
bool operator <= (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.compare(rhs) <= 0;
}

template<typename BidiIter>
bool operator >= (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.compare(rhs) >= 0;
}

template<typename BidiIter>
bool operator > (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.compare(rhs) > 0;
}

template<typename BidiIter>
bool operator == (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs == rhs.str();
}

template<typename BidiIter>
bool operator != (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs != rhs.str();
}

template<typename BidiIter>
bool operator < (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs < rhs.str();
}

template<typename BidiIter>
bool operator > (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs> rhs.str();
}

template<typename BidiIter>
bool operator >= (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs >= rhs.str();
}

template<typename BidiIter>
bool operator <= (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs <= rhs.str();
}

template<typename BidiIter>
bool operator == (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() == rhs;
}

template<typename BidiIter>
bool operator != (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() != rhs;
}

template<typename BidiIter>
bool operator < (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() < rhs;
}

template<typename BidiIter>
bool operator > (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() > rhs;
}

template<typename BidiIter>
bool operator >= (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() >= rhs;
}

template<typename BidiIter>
bool operator <= (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() <= rhs;
}

template<typename BidiIter>
bool operator == (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs == rhs.str();
}

template<typename BidiIter>
bool operator != (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs != rhs.str();
}

template<typename BidiIter>
bool operator < (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs < rhs.str();
}

template<typename BidiIter>
bool operator > (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs> rhs.str();
}

template<typename BidiIter>
bool operator >= (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs >= rhs.str();
}

template<typename BidiIter>
bool operator <= (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs <= rhs.str();
}

template<typename BidiIter>
bool operator == (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() == rhs;
}

template<typename BidiIter>
bool operator != (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() != rhs;
}

template<typename BidiIter>
bool operator < (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() < rhs;
}

template<typename BidiIter>
bool operator > (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() > rhs;
}

template<typename BidiIter>
bool operator >= (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() >= rhs;
}

template<typename BidiIter>
bool operator <= (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() <= rhs;
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (sub_match<BidiIter> const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs.str() + rhs.str();
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const &rhs)
{
return lhs.str() + rhs;
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (typename iterator_value<BidiIter>::type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs + rhs.str();
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (sub_match<BidiIter> const &lhs, typename iterator_value<BidiIter>::type const *rhs)
{
return lhs.str() + rhs;
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (typename iterator_value<BidiIter>::type const *lhs, sub_match<BidiIter> const &rhs)
{
return lhs + rhs.str();
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (sub_match<BidiIter> const &lhs, typename sub_match<BidiIter>::string_type const &rhs)
{
return lhs.str() + rhs;
}

template<typename BidiIter>
typename sub_match<BidiIter>::string_type
operator + (typename sub_match<BidiIter>::string_type const &lhs, sub_match<BidiIter> const &rhs)
{
return lhs + rhs.str();
}

}} 

namespace boost
{
template<typename BidiIter>
struct range_mutable_iterator<xpressive::sub_match<BidiIter> >
{
typedef BidiIter type;
};

template<typename BidiIter>
struct range_const_iterator<xpressive::sub_match<BidiIter> >
{
typedef BidiIter type;
};
}

#endif
