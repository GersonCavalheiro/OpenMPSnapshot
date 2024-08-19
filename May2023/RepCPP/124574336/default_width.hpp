
#if !defined(BOOST_SPIRIT_KARMA_DEFAULT_WIDTH_APR_07_2009_0912PM)
#define BOOST_SPIRIT_KARMA_DEFAULT_WIDTH_APR_07_2009_0912PM

#if defined(_MSC_VER)
#pragma once
#endif

#if !defined(BOOST_KARMA_DEFAULT_FIELD_LENGTH)
#define BOOST_KARMA_DEFAULT_FIELD_LENGTH 10
#endif

#if !defined(BOOST_KARMA_DEFAULT_FIELD_MAXWIDTH)
#define BOOST_KARMA_DEFAULT_FIELD_MAXWIDTH 10
#endif

#if !defined(BOOST_KARMA_DEFAULT_COLUMNS)
#define BOOST_KARMA_DEFAULT_COLUMNS 5
#endif

namespace boost { namespace spirit { namespace karma { namespace detail
{
struct default_width
{
operator int() const
{
return BOOST_KARMA_DEFAULT_FIELD_LENGTH;
}
};

struct default_max_width
{
operator int() const
{
return BOOST_KARMA_DEFAULT_FIELD_MAXWIDTH;
}
};

struct default_columns
{
operator int() const
{
return BOOST_KARMA_DEFAULT_COLUMNS;
}
};

}}}}

#endif
