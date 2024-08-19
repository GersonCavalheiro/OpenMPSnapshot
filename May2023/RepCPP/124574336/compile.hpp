
#ifndef BOOST_XPRESSIVE_DETAIL_STATIC_COMPILE_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_STATIC_COMPILE_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/proto/core.hpp>
#include <boost/xpressive/regex_traits.hpp>
#include <boost/xpressive/detail/core/regex_impl.hpp>
#include <boost/xpressive/detail/core/linker.hpp>
#include <boost/xpressive/detail/core/optimize.hpp>
#include <boost/xpressive/detail/core/adaptor.hpp>
#include <boost/xpressive/detail/core/matcher/end_matcher.hpp>
#include <boost/xpressive/detail/static/static.hpp>
#include <boost/xpressive/detail/static/visitor.hpp>
#include <boost/xpressive/detail/static/grammar.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Xpr, typename BidiIter, typename Traits>
void static_compile_impl2(Xpr const &xpr, shared_ptr<regex_impl<BidiIter> > const &impl, Traits const &tr)
{
typedef typename iterator_value<BidiIter>::type char_type;
impl->tracking_clear();
impl->traits_ = new traits_holder<Traits>(tr);

typedef xpression_visitor<BidiIter, mpl::false_, Traits> visitor_type;
visitor_type visitor(tr, impl);
intrusive_ptr<matchable_ex<BidiIter> const> adxpr = make_adaptor<matchable_ex<BidiIter> >(
typename Grammar<char_type>::template impl<Xpr const &, end_xpression, visitor_type &>()(
xpr
, end_xpression()
, visitor
)
);

common_compile(adxpr, *impl, visitor.traits());

impl->tracking_update();
}

struct XpressiveLocaleModifier
: proto::binary_expr<
modifier_tag
, proto::terminal<locale_modifier<proto::_> >
, proto::_
>
{};

template<typename Xpr, typename BidiIter>
typename disable_if<proto::matches<Xpr, XpressiveLocaleModifier> >::type
static_compile_impl1(Xpr const &xpr, shared_ptr<regex_impl<BidiIter> > const &impl)
{
typedef typename iterator_value<BidiIter>::type char_type;
typedef typename default_regex_traits<char_type>::type traits_type;
traits_type tr;
static_compile_impl2(xpr, impl, tr);
}

template<typename Xpr, typename BidiIter>
typename enable_if<proto::matches<Xpr, XpressiveLocaleModifier> >::type
static_compile_impl1(Xpr const &xpr, shared_ptr<regex_impl<BidiIter> > const &impl)
{
typedef typename proto::result_of::value<typename proto::result_of::left<Xpr>::type>::type::locale_type locale_type;
typedef typename regex_traits_type<locale_type, BidiIter>::type traits_type;
static_compile_impl2(proto::right(xpr), impl, traits_type(proto::value(proto::left(xpr)).getloc()));
}

template<typename Xpr, typename BidiIter>
void static_compile(Xpr const &xpr, shared_ptr<regex_impl<BidiIter> > const &impl)
{
static_compile_impl1(xpr, impl);
}

}}} 

#endif
