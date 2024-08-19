
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_OPTIONAL_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_OPTIONAL_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Xpr, typename Greedy>
struct optional_matcher
: quant_style<quant_variable_width, unknown_width::value, Xpr::pure>
{
Xpr xpr_;

explicit optional_matcher(Xpr const &xpr)
: xpr_(xpr)
{
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
return this->match_(state, next, Greedy());
}

private:
template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::true_) const 
{
return this->xpr_.BOOST_NESTED_TEMPLATE push_match<Next>(state)
|| next.match(state);
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::false_) const 
{
return next.match(state)
|| this->xpr_.BOOST_NESTED_TEMPLATE push_match<Next>(state);
}

optional_matcher &operator =(optional_matcher const &);
};

template<typename BidiIter, typename Next>
inline bool match_next(match_state<BidiIter> &state, Next const &next, int mark_number)
{
sub_match_impl<BidiIter> &br = state.sub_match(mark_number);

bool old_matched = br.matched;
br.matched = false;

if(next.match(state))
{
return true;
}

br.matched = old_matched;
return false;
}

template<typename Xpr, typename Greedy>
struct optional_mark_matcher
: quant_style<quant_variable_width, unknown_width::value, Xpr::pure>
{
Xpr xpr_;
int mark_number_;

explicit optional_mark_matcher(Xpr const &xpr, int mark_number)
: xpr_(xpr)
, mark_number_(mark_number)
{
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
return this->match_(state, next, Greedy());
}

private:
template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::true_) const 
{
return this->xpr_.BOOST_NESTED_TEMPLATE push_match<Next>(state)
|| match_next(state, next, this->mark_number_);
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::false_) const 
{
return match_next(state, next, this->mark_number_)
|| this->xpr_.BOOST_NESTED_TEMPLATE push_match<Next>(state);
}

optional_mark_matcher &operator =(optional_mark_matcher const &);
};

}}}

#endif
