
#ifndef BOOST_XPRESSIVE_DETAIL_STATIC_PLACEHOLDERS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_STATIC_PLACEHOLDERS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
# pragma warning(push)
# pragma warning(disable:4510) 
# pragma warning(disable:4610) 
#endif

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/regex_impl.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct mark_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_variable_width, unknown_width::value, true)

int mark_number_;
};

struct posix_charset_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_fixed_width, 1, true)

char const *name_;
bool not_;
};

template<typename Cond>
struct assert_word_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_none, 0, true)
};

template<typename Char>
struct range_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_fixed_width, 1, true)

Char ch_min_;
Char ch_max_;
bool not_;
};

struct assert_bol_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_none, 0, true)
};

struct assert_eol_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_none, 0, true)
};

struct logical_newline_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_variable_width, unknown_width::value, true)
};

struct self_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_variable_width, unknown_width::value, false)
};

template<typename Nbr>
struct attribute_placeholder
{
BOOST_XPR_QUANT_STYLE(quant_variable_width, unknown_width::value, false)

typedef Nbr nbr_type;
static Nbr nbr() { return Nbr(); }
};

}}} 

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
