
#ifndef BOOST_XPRESSIVE_DETAIL_SET_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_SET_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
# pragma warning(push)
# pragma warning(disable : 4127) 
# pragma warning(disable : 4100) 
# pragma warning(disable : 4351) 
#endif

#include <algorithm>
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/same_traits.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Traits, typename Size>
struct set_matcher
: quant_style_fixed_width<1>
{
typedef typename Traits::char_type char_type;
char_type set_[ Size::value ];
bool not_;
bool icase_;

set_matcher()
: set_()
, not_(false)
, icase_(false)
{
}

void inverse()
{
this->not_ = !this->not_;
}

void nocase(Traits const &tr)
{
this->icase_ = true;

for(int i = 0; i < Size::value; ++i)
{
this->set_[i] = tr.translate_nocase(this->set_[i]);
}
}

bool in_set(Traits const &tr, char_type ch) const
{
char_type const *begin = &this->set_[0], *end = begin + Size::value;
ch = this->icase_ ? tr.translate_nocase(ch) : tr.translate(ch);
return end != std::find(begin, end, ch);
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
if(state.eos() || this->not_ == this->in_set(traits_cast<Traits>(state), *state.cur_))
{
return false;
}

if(++state.cur_, next.match(state))
{
return true;
}

return --state.cur_, false;
}
};

struct set_initializer
{
};

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

}}} 

#endif
