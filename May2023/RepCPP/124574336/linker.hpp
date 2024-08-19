
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_LINKER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_LINKER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#ifndef BOOST_NO_STD_LOCALE
# include <locale>
#endif
#include <stack>
#include <limits>
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/version.hpp>

#if BOOST_VERSION >= 103500
# include <boost/fusion/include/for_each.hpp>
#else
# include <boost/spirit/fusion/algorithm/for_each.hpp>
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/dynamic/matchable.hpp>
#include <boost/xpressive/detail/core/matchers.hpp>
#include <boost/xpressive/detail/core/peeker.hpp>
#include <boost/xpressive/detail/utility/never_true.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct icase_modifier
{
template<typename Visitor>
struct apply {};

template<typename BidiIter, typename ICase, typename Traits>
struct apply<xpression_visitor<BidiIter, ICase, Traits> >
{
typedef xpression_visitor<BidiIter, mpl::true_, Traits> type;
};

template<typename Visitor>
static typename apply<Visitor>::type
call(Visitor &visitor)
{
return typename apply<Visitor>::type(visitor.traits(), visitor.self());
}
};

template<typename Locale, typename BidiIter>
struct regex_traits_type
{
#ifndef BOOST_NO_STD_LOCALE

typedef typename iterator_value<BidiIter>::type char_type;

typedef typename mpl::if_c
<
is_same<Locale, std::locale>::value
, cpp_regex_traits<char_type>
, Locale
>::type type;

#else

typedef Locale type;

#endif
};

template<typename Locale>
struct locale_modifier
{
typedef Locale locale_type;

locale_modifier(Locale const &loc)
: loc_(loc)
{
}

template<typename Visitor>
struct apply {};

template<typename BidiIter, typename ICase, typename OtherTraits>
struct apply<xpression_visitor<BidiIter, ICase, OtherTraits> >
{
typedef typename regex_traits_type<Locale, BidiIter>::type traits_type;
typedef xpression_visitor<BidiIter, ICase, traits_type> type;
};

template<typename Visitor>
typename apply<Visitor>::type
call(Visitor &visitor) const
{
return typename apply<Visitor>::type(this->loc_, visitor.self());
}

Locale getloc() const
{
return this->loc_;
}

private:
Locale loc_;
};

template<typename Char>
struct xpression_linker
{
template<typename Traits>
explicit xpression_linker(Traits const &tr)
: back_stack_()
, traits_(&tr)
, traits_type_(&typeid(Traits))
, has_backrefs_(false)
{
}

template<typename Matcher>
void accept(Matcher const &, void const *)
{
}

template<typename Traits, typename ICase>
void accept(mark_matcher<Traits, ICase> const &, void const *)
{
this->has_backrefs_ = true;
}

template<typename Action>
void accept(action_matcher<Action> const &, void const *)
{
this->has_backrefs_ = true;
}

template<typename Predicate>
void accept(predicate_matcher<Predicate> const &, void const *)
{
this->has_backrefs_ = true;
}

void accept(repeat_begin_matcher const &, void const *next)
{
this->back_stack_.push(next);
}

template<typename Greedy>
void accept(repeat_end_matcher<Greedy> const &matcher, void const *)
{
matcher.back_ = this->back_stack_.top();
this->back_stack_.pop();
}

template<typename Alternates, typename Traits>
void accept(alternate_matcher<Alternates, Traits> const &matcher, void const *next)
{
xpression_peeker<Char> peeker(matcher.bset_, this->get_traits<Traits>());
this->alt_link(matcher.alternates_, next, &peeker);
}

void accept(alternate_end_matcher const &matcher, void const *)
{
matcher.back_ = this->back_stack_.top();
this->back_stack_.pop();
}

template<typename Xpr, typename Greedy>
void accept(optional_matcher<Xpr, Greedy> const &matcher, void const *next)
{
this->back_stack_.push(next);
matcher.xpr_.link(*this);
}

template<typename Xpr, typename Greedy>
void accept(optional_mark_matcher<Xpr, Greedy> const &matcher, void const *next)
{
this->back_stack_.push(next);
matcher.xpr_.link(*this);
}

template<typename Xpr>
void accept(keeper_matcher<Xpr> const &matcher, void const *)
{
matcher.xpr_.link(*this);
}

template<typename Xpr>
void accept(lookahead_matcher<Xpr> const &matcher, void const *)
{
matcher.xpr_.link(*this);
}

template<typename Xpr>
void accept(lookbehind_matcher<Xpr> const &matcher, void const *)
{
matcher.xpr_.link(*this);
}

template<typename Xpr, typename Greedy>
void accept(simple_repeat_matcher<Xpr, Greedy> const &matcher, void const *)
{
matcher.xpr_.link(*this);
}

bool has_backrefs() const
{
return this->has_backrefs_;
}

template<typename Xpr>
void alt_branch_link(Xpr const &xpr, void const *next, xpression_peeker<Char> *peeker)
{
this->back_stack_.push(next);
xpr.link(*this);
xpr.peek(*peeker);
}

private:

struct alt_link_pred
{
xpression_linker<Char> *linker_;
xpression_peeker<Char> *peeker_;
void const *next_;

alt_link_pred
(
xpression_linker<Char> *linker
, xpression_peeker<Char> *peeker
, void const *next
)
: linker_(linker)
, peeker_(peeker)
, next_(next)
{
}

template<typename Xpr>
void operator ()(Xpr const &xpr) const
{
this->linker_->alt_branch_link(xpr, this->next_, this->peeker_);
}
};

template<typename BidiIter>
void alt_link
(
alternates_vector<BidiIter> const &alternates
, void const *next
, xpression_peeker<Char> *peeker
)
{
std::for_each(alternates.begin(), alternates.end(), alt_link_pred(this, peeker, next));
}

template<typename Alternates>
void alt_link
(
fusion::sequence_base<Alternates> const &alternates
, void const *next
, xpression_peeker<Char> *peeker
)
{
#if BOOST_VERSION >= 103500
fusion::for_each(alternates.derived(), alt_link_pred(this, peeker, next));
#else
fusion::for_each(alternates.cast(), alt_link_pred(this, peeker, next));
#endif
}

template<typename Traits>
Traits const &get_traits() const
{
BOOST_ASSERT(*this->traits_type_ == typeid(Traits));
return *static_cast<Traits const *>(this->traits_);
}

std::stack<void const *> back_stack_;
void const *traits_;
std::type_info const *traits_type_;
bool has_backrefs_;
};

}}} 

#endif
