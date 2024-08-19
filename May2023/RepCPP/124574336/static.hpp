
#ifndef BOOST_XPRESSIVE_DETAIL_STATIC_STATIC_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_STATIC_STATIC_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/assert.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/core/linker.hpp>
#include <boost/xpressive/detail/core/peeker.hpp>
#include <boost/xpressive/detail/static/placeholders.hpp>
#include <boost/xpressive/detail/utility/width.hpp>


namespace boost { namespace xpressive { namespace detail
{

template<typename Top, typename Next>
struct stacked_xpression
: Next
{
template<typename BidiIter>
bool match(match_state<BidiIter> &state) const
{
return static_cast<Next const *>(this)->
BOOST_NESTED_TEMPLATE push_match<Top>(state);
}

template<typename BidiIter>
static bool top_match(match_state<BidiIter> &state, void const *top)
{
return static_cast<Top const *>(top)->
BOOST_NESTED_TEMPLATE push_match<Top>(state);
}

template<typename BidiIter>
static bool pop_match(match_state<BidiIter> &state, void const *top)
{
return static_cast<Top const *>(top)->match(state);
}

template<typename BidiIter>
bool skip_match(match_state<BidiIter> &state) const
{
return Top::skip_impl(*static_cast<Next const *>(this), state);
}


template<typename That, typename BidiIter>
static bool skip_impl(That const &that, match_state<BidiIter> &state)
{
return that.BOOST_NESTED_TEMPLATE push_match<Top>(state);
}
};

template<typename Top, typename Next>
inline stacked_xpression<Top, Next> const &stacked_xpression_cast(Next const &next)
{
BOOST_MPL_ASSERT_RELATION(sizeof(stacked_xpression<Top, Next>), ==, sizeof(Next));
return *static_cast<stacked_xpression<Top, Next> const *>(&next);
}

template<typename Matcher, typename Next>
struct static_xpression
: Matcher
{
Next next_;

BOOST_STATIC_CONSTANT(bool, pure = Matcher::pure && Next::pure);
BOOST_STATIC_CONSTANT(
std::size_t
, width =
Matcher::width != unknown_width::value && Next::width != unknown_width::value
? Matcher::width + Next::width
: unknown_width::value
);

static_xpression(Matcher const &matcher = Matcher(), Next const &next = Next())
: Matcher(matcher)
, next_(next)
{
}

template<typename BidiIter>
bool match(match_state<BidiIter> &state) const
{
return this->Matcher::match(state, this->next_);
}

template<typename Top, typename BidiIter>
bool push_match(match_state<BidiIter> &state) const
{
return this->Matcher::match(state, stacked_xpression_cast<Top>(this->next_));
}

template<typename That, typename BidiIter>
static bool skip_impl(That const &that, match_state<BidiIter> &state)
{
return that.match(state);
}

template<typename Char>
void link(xpression_linker<Char> &linker) const
{
linker.accept(*static_cast<Matcher const *>(this), &this->next_);
this->next_.link(linker);
}

template<typename Char>
void peek(xpression_peeker<Char> &peeker) const
{
this->peek_next_(peeker.accept(*static_cast<Matcher const *>(this)), peeker);
}

detail::width get_width() const
{
return this->get_width_(mpl::size_t<width>());
}

private:

static_xpression &operator =(static_xpression const &);

template<typename Char>
void peek_next_(mpl::true_, xpression_peeker<Char> &peeker) const
{
this->next_.peek(peeker);
}

template<typename Char>
void peek_next_(mpl::false_, xpression_peeker<Char> &) const
{
}

template<std::size_t Width>
detail::width get_width_(mpl::size_t<Width>) const
{
return Width;
}

detail::width get_width_(unknown_width) const
{
return this->Matcher::get_width() + this->next_.get_width();
}
};

template<typename Matcher>
inline static_xpression<Matcher> const
make_static(Matcher const &matcher)
{
return static_xpression<Matcher>(matcher);
}

template<typename Matcher, typename Next>
inline static_xpression<Matcher, Next> const
make_static(Matcher const &matcher, Next const &next)
{
return static_xpression<Matcher, Next>(matcher, next);
}

struct no_next
{
BOOST_STATIC_CONSTANT(std::size_t, width = 0);
BOOST_STATIC_CONSTANT(bool, pure = true);

template<typename Char>
void link(xpression_linker<Char> &) const
{
}

template<typename Char>
void peek(xpression_peeker<Char> &peeker) const
{
peeker.fail();
}

detail::width get_width() const
{
return 0;
}
};

inline int get_mark_number(basic_mark_tag const &mark)
{
return proto::value(mark).mark_number_;
}

}}} 

#endif
