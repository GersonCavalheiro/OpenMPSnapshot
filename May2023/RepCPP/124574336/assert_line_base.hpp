
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_DETAIL_ASSERT_LINE_BASE_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_DETAIL_ASSERT_LINE_BASE_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Traits>
struct assert_line_base
: quant_style_assertion
{
typedef typename Traits::char_type char_type;
typedef typename Traits::char_class_type char_class_type;

protected:
assert_line_base(Traits const &tr)
: newline_(lookup_classname(tr, "newline"))
, nl_(tr.widen('\n'))
, cr_(tr.widen('\r'))
{
}

char_class_type newline_;
char_type nl_, cr_;
};

}}}

#endif
