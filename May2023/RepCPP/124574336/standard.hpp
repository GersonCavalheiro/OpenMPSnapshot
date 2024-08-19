
#if !defined(BOOST_SPIRIT_STANDARD_APRIL_26_2006_1106PM)
#define BOOST_SPIRIT_STANDARD_APRIL_26_2006_1106PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <cctype>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

namespace boost { namespace spirit { namespace char_encoding
{
struct standard
{
typedef char char_type;
typedef unsigned char classify_type;

static bool
isascii_(int ch)
{
return 0 == (ch & ~0x7f);
}

static bool
ischar(int ch)
{
return (0 == (ch & ~0xff) || ~0 == (ch | 0xff)) != 0;
}


static bool
strict_ischar(int ch)
{
return ch >= 0 && ch <= UCHAR_MAX;
}

static bool
isalnum(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isalnum(ch) != 0;
}

static bool
isalpha(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isalpha(ch) != 0;
}

static bool
isdigit(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isdigit(ch) != 0;
}

static bool
isxdigit(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isxdigit(ch) != 0;
}

static bool
iscntrl(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::iscntrl(ch) != 0;
}

static bool
isgraph(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isgraph(ch) != 0;
}

static bool
islower(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::islower(ch) != 0;
}

static bool
isprint(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isprint(ch) != 0;
}

static bool
ispunct(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::ispunct(ch) != 0;
}

static bool
isspace(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isspace(ch) != 0;
}

static bool
isblank BOOST_PREVENT_MACRO_SUBSTITUTION (int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ch == ' ' || ch == '\t');
}

static bool
isupper(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::isupper(ch) != 0;
}


static int
tolower(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::tolower(ch);
}

static int
toupper(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return std::toupper(ch);
}

static ::boost::uint32_t
toucs4(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return ch;
}
};
}}}

#endif
