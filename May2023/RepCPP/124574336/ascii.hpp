
#if !defined(BOOST_SPIRIT_ASCII_APRIL_26_2006_1106PM)
#define BOOST_SPIRIT_ASCII_APRIL_26_2006_1106PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <climits>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>

#define BOOST_CC_DIGIT    0x0001
#define BOOST_CC_XDIGIT   0x0002
#define BOOST_CC_ALPHA    0x0004
#define BOOST_CC_CTRL     0x0008
#define BOOST_CC_LOWER    0x0010
#define BOOST_CC_UPPER    0x0020
#define BOOST_CC_SPACE    0x0040
#define BOOST_CC_PUNCT    0x0080

namespace boost { namespace spirit { namespace char_encoding
{

const unsigned char ascii_char_types[] =
{
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL|BOOST_CC_SPACE,
BOOST_CC_CTRL|BOOST_CC_SPACE,
BOOST_CC_CTRL|BOOST_CC_SPACE,
BOOST_CC_CTRL|BOOST_CC_SPACE,
BOOST_CC_CTRL|BOOST_CC_SPACE,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_CTRL,
BOOST_CC_SPACE,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_DIGIT|BOOST_CC_XDIGIT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_ALPHA|BOOST_CC_UPPER,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_XDIGIT|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_ALPHA|BOOST_CC_LOWER,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_PUNCT,
BOOST_CC_CTRL,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

struct ascii
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
return isascii_(ch);
}


static bool
strict_ischar(int ch)
{
return ch >= 0 && ch <= 127;
}

static bool
isalnum(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_ALPHA)
|| (ascii_char_types[ch] & BOOST_CC_DIGIT);
}

static bool
isalpha(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_ALPHA) ? true : false;
}

static bool
isdigit(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_DIGIT) ? true : false;
}

static bool
isxdigit(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_XDIGIT) ? true : false;
}

static bool
iscntrl(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_CTRL) ? true : false;
}

static bool
isgraph(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return ('\x21' <= ch && ch <= '\x7e');
}

static bool
islower(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_LOWER) ? true : false;
}

static bool
isprint(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return ('\x20' <= ch && ch <= '\x7e');
}

static bool
ispunct(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_PUNCT) ? true : false;
}

static bool
isspace(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_SPACE) ? true : false;
}

static bool
isblank BOOST_PREVENT_MACRO_SUBSTITUTION (int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return ('\x09' == ch || '\x20' == ch);
}

static bool
isupper(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return (ascii_char_types[ch] & BOOST_CC_UPPER) ? true : false;
}


static int
tolower(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return isupper(ch) ? (ch - 'A' + 'a') : ch;
}

static int
toupper(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return islower(ch) ? (ch - 'a' + 'A') : ch;
}

static ::boost::uint32_t
toucs4(int ch)
{
BOOST_ASSERT(strict_ischar(ch));
return ch;
}
};

}}}

#undef BOOST_CC_DIGIT
#undef BOOST_CC_XDIGIT
#undef BOOST_CC_ALPHA
#undef BOOST_CC_CTRL
#undef BOOST_CC_LOWER
#undef BOOST_CC_UPPER
#undef BOOST_CC_PUNCT
#undef BOOST_CC_SPACE

#endif
