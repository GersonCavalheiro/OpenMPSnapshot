
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ASSERT_WORD_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ASSERT_WORD_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/utility/ignore_unused.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename IsBoundary>
struct word_boundary
{
template<typename BidiIter>
static bool eval(bool prevword, bool thisword, match_state<BidiIter> &state)
{
if((state.flags_.match_not_bow_ && state.bos()) || (state.flags_.match_not_eow_ && state.eos()))
{
return !IsBoundary::value;
}

return IsBoundary::value == (prevword != thisword);
}
};

struct word_begin
{
template<typename BidiIter>
static bool eval(bool prevword, bool thisword, match_state<BidiIter> &state)
{
if(state.flags_.match_not_bow_ && state.bos())
{
return false;
}

return !prevword && thisword;
}
};

struct word_end
{
template<typename BidiIter>
static bool eval(bool prevword, bool thisword, match_state<BidiIter> &state)
{
if(state.flags_.match_not_eow_ && state.eos())
{
return false;
}

return prevword && !thisword;
}
};

template<typename Cond, typename Traits>
struct assert_word_matcher
: quant_style_assertion
{
typedef typename Traits::char_type char_type;
typedef typename Traits::char_class_type char_class_type;

assert_word_matcher(Traits const &tr)
: word_(lookup_classname(tr, "w"))
{
BOOST_ASSERT(0 != this->word_);
}

assert_word_matcher(char_class_type word)
: word_(word)
{}

bool is_word(Traits const &tr, char_type ch) const
{
detail::ignore_unused(tr);
return tr.isctype(tr.translate(ch), this->word_);
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
BidiIter cur = state.cur_;
bool const thisword = !state.eos() && this->is_word(traits_cast<Traits>(state), *cur);
bool const prevword = (!state.bos() || state.flags_.match_prev_avail_)
&& this->is_word(traits_cast<Traits>(state), *--cur);

return Cond::eval(prevword, thisword, state) && next.match(state);
}

char_class_type word() const
{
return this->word_;
}

private:
char_class_type word_;
};

}}}

#endif
