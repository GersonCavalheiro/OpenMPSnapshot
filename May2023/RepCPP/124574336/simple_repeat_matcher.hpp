
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_SIMPLE_REPEAT_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_SIMPLE_REPEAT_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/next_prior.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/static/type_traits.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct greedy_slow_tag {};
struct greedy_fast_tag {};
struct non_greedy_tag {};

typedef static_xpression<any_matcher, true_xpression> any_sxpr;
typedef matcher_wrapper<any_matcher> any_dxpr;

template<typename Xpr, typename Greedy, typename Random>
struct simple_repeat_traits
{
typedef typename mpl::if_c<Greedy::value, greedy_slow_tag, non_greedy_tag>::type tag_type;
};

template<>
struct simple_repeat_traits<any_sxpr, mpl::true_, mpl::true_>
{
typedef greedy_fast_tag tag_type;
};

template<>
struct simple_repeat_traits<any_dxpr, mpl::true_, mpl::true_>
{
typedef greedy_fast_tag tag_type;
};

template<typename Xpr, typename Greedy>
struct simple_repeat_matcher
: quant_style_variable_width
{
typedef Xpr xpr_type;
typedef Greedy greedy_type;

Xpr xpr_;
unsigned int min_, max_;
std::size_t width_;
mutable bool leading_;

simple_repeat_matcher(Xpr const &xpr, unsigned int min, unsigned int max, std::size_t width)
: xpr_(xpr)
, min_(min)
, max_(max)
, width_(width)
, leading_(false)
{
BOOST_ASSERT(min <= max);
BOOST_ASSERT(0 != max);
BOOST_ASSERT(0 != width && unknown_width() != width);
BOOST_ASSERT(Xpr::width == unknown_width() || Xpr::width == width);
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
typedef mpl::bool_<is_random<BidiIter>::value> is_rand;
typedef typename simple_repeat_traits<Xpr, greedy_type, is_rand>::tag_type tag_type;
return this->match_(state, next, tag_type());
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, greedy_slow_tag) const
{
int const diff = -static_cast<int>(Xpr::width == unknown_width::value ? this->width_ : Xpr::width);
unsigned int matches = 0;
BidiIter const tmp = state.cur_;

while(matches < this->max_ && this->xpr_.match(state))
{
++matches;
}

if(this->leading_)
{
state.next_search_ = (matches && matches < this->max_)
? state.cur_
: (tmp == state.end_) ? tmp : boost::next(tmp);
}

if(this->min_ > matches)
{
state.cur_ = tmp;
return false;
}

for(; ; --matches, std::advance(state.cur_, diff))
{
if(next.match(state))
{
return true;
}
else if(this->min_ == matches)
{
state.cur_ = tmp;
return false;
}
}
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, non_greedy_tag) const
{
BOOST_ASSERT(!this->leading_);
BidiIter const tmp = state.cur_;
unsigned int matches = 0;

for(; matches < this->min_; ++matches)
{
if(!this->xpr_.match(state))
{
state.cur_ = tmp;
return false;
}
}

do
{
if(next.match(state))
{
return true;
}
}
while(matches++ < this->max_ && this->xpr_.match(state));

state.cur_ = tmp;
return false;
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, greedy_fast_tag) const
{
BidiIter const tmp = state.cur_;
std::size_t const diff_to_end = static_cast<std::size_t>(state.end_ - tmp);

if(this->min_ > diff_to_end)
{
if(this->leading_)
{
state.next_search_ = (tmp == state.end_) ? tmp : boost::next(tmp);
}
return false;
}

BidiIter const min_iter = tmp + this->min_;
state.cur_ += (std::min)((std::size_t)this->max_, diff_to_end);

if(this->leading_)
{
state.next_search_ = (diff_to_end && diff_to_end < this->max_)
? state.cur_
: (tmp == state.end_) ? tmp : boost::next(tmp);
}

for(;; --state.cur_)
{
if(next.match(state))
{
return true;
}
else if(min_iter == state.cur_)
{
state.cur_ = tmp;
return false;
}
}
}

detail::width get_width() const
{
if(this->min_ != this->max_)
{
return unknown_width::value;
}
return this->min_ * this->width_;
}

private:
simple_repeat_matcher &operator =(simple_repeat_matcher const &);
};



}}}

#endif
