
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_QUANT_STYLE_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_QUANT_STYLE_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/mpl/has_xxx.hpp>
#include <boost/xpressive/detail/utility/width.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>

namespace boost { namespace xpressive { namespace detail
{

BOOST_MPL_HAS_XXX_TRAIT_DEF(is_boost_xpressive_xpression_)

template<typename Xpr>
struct is_xpr
: has_is_boost_xpressive_xpression_<Xpr>
{};

enum quant_enum
{
quant_none,
quant_fixed_width,
quant_variable_width
};

template<quant_enum QuantStyle, std::size_t Width = unknown_width::value, bool Pure = true>
struct quant_style
{
typedef void is_boost_xpressive_xpression_;

BOOST_STATIC_CONSTANT(int, quant = QuantStyle);

BOOST_STATIC_CONSTANT(std::size_t, width = Width);

BOOST_STATIC_CONSTANT(bool, pure = Pure);

static detail::width get_width()
{
return width;
}
};

#define BOOST_XPR_QUANT_STYLE(Style, Width, Pure)                               \
typedef void is_boost_xpressive_xpression_;                                 \
BOOST_STATIC_CONSTANT(int, quant = Style);                                  \
BOOST_STATIC_CONSTANT(std::size_t, width = Width);                          \
BOOST_STATIC_CONSTANT(bool, pure = Pure);                                   \
static detail::width get_width() { return width; }                          \



typedef quant_style<quant_none> quant_style_none;

typedef quant_style<quant_fixed_width> quant_style_fixed_unknown_width;

typedef quant_style<quant_variable_width> quant_style_variable_width;

template<std::size_t Width>
struct quant_style_fixed_width
: quant_style<quant_fixed_width, Width>
{
};

struct quant_style_assertion
: quant_style<quant_none, 0>
{
};

template<typename Matcher>
struct quant_type
: mpl::int_<Matcher::quant>
{
};

}}} 

#endif
