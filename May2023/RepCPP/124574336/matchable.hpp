
#ifndef BOOST_XPRESSIVE_DETAIL_DYNAMIC_MATCHABLE_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_DYNAMIC_MATCHABLE_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/utility/counted_base.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/dynamic/sequence.hpp>
#include <boost/xpressive/regex_error.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct quant_spec
{
unsigned int min_;
unsigned int max_;
bool greedy_;
std::size_t *hidden_mark_count_;
};

template<typename BidiIter>
struct matchable
{
typedef BidiIter iterator_type;
typedef typename iterator_value<iterator_type>::type char_type;
virtual ~matchable() {}
virtual bool match(match_state<BidiIter> &state) const = 0;
};

template<typename BidiIter>
struct matchable_ex
: matchable<BidiIter>
, counted_base<matchable_ex<BidiIter> >
{
typedef BidiIter iterator_type;
typedef typename iterator_value<iterator_type>::type char_type;

virtual void link(xpression_linker<char_type> &) const
{
}

virtual void peek(xpression_peeker<char_type> &peeker) const
{
peeker.fail();
}

virtual void repeat(quant_spec const &, sequence<BidiIter> &) const
{
BOOST_THROW_EXCEPTION(
regex_error(regex_constants::error_badrepeat, "expression cannot be quantified")
);
}


template<typename Top>
bool push_match(match_state<BidiIter> &state) const
{
BOOST_MPL_ASSERT((is_same<Top, matchable_ex<BidiIter> >));
return this->match(state);
}

static bool top_match(match_state<BidiIter> &state, void const *top)
{
return static_cast<matchable_ex<BidiIter> const *>(top)->match(state);
}

static bool pop_match(match_state<BidiIter> &state, void const *top)
{
return static_cast<matchable_ex<BidiIter> const *>(top)->match(state);
}

bool skip_match(match_state<BidiIter> &state) const
{
return this->match(state);
}
};

template<typename BidiIter>
struct shared_matchable
{
typedef BidiIter iterator_type;
typedef typename iterator_value<BidiIter>::type char_type;
typedef intrusive_ptr<matchable_ex<BidiIter> const> matchable_ptr;

BOOST_STATIC_CONSTANT(std::size_t, width = unknown_width::value);
BOOST_STATIC_CONSTANT(bool, pure = false);

shared_matchable(matchable_ptr const &xpr = matchable_ptr())
: xpr_(xpr)
{
}

bool operator !() const
{
return !this->xpr_;
}

friend bool operator ==(shared_matchable<BidiIter> const &left, shared_matchable<BidiIter> const &right)
{
return left.xpr_ == right.xpr_;
}

friend bool operator !=(shared_matchable<BidiIter> const &left, shared_matchable<BidiIter> const &right)
{
return left.xpr_ != right.xpr_;
}

matchable_ptr const &matchable() const
{
return this->xpr_;
}

bool match(match_state<BidiIter> &state) const
{
return this->xpr_->match(state);
}

void link(xpression_linker<char_type> &linker) const
{
this->xpr_->link(linker);
}

void peek(xpression_peeker<char_type> &peeker) const
{
this->xpr_->peek(peeker);
}

template<typename Top>
bool push_match(match_state<BidiIter> &state) const
{
BOOST_MPL_ASSERT((is_same<Top, matchable_ex<BidiIter> >));
return this->match(state);
}

private:
matchable_ptr xpr_;
};

}}} 

#endif
