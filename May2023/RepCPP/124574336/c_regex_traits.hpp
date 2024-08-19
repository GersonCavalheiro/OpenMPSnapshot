
#ifndef BOOST_XPRESSIVE_TRAITS_C_REGEX_TRAITS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_TRAITS_C_REGEX_TRAITS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <cstdlib>
#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/xpressive/traits/detail/c_ctype.hpp>

namespace boost { namespace xpressive
{

namespace detail
{
struct empty_locale
{
};

template<typename Char, std::size_t SizeOfChar = sizeof(Char)>
struct c_regex_traits_base
{
protected:
template<typename Traits>
void imbue(Traits const &tr)
{
}
};

template<typename Char>
struct c_regex_traits_base<Char, 1>
{
protected:
template<typename Traits>
static void imbue(Traits const &)
{
}
};

#ifndef BOOST_XPRESSIVE_NO_WREGEX
template<std::size_t SizeOfChar>
struct c_regex_traits_base<wchar_t, SizeOfChar>
{
protected:
template<typename Traits>
static void imbue(Traits const &)
{
}
};
#endif

template<typename Char>
Char c_tolower(Char);

template<typename Char>
Char c_toupper(Char);

template<>
inline char c_tolower(char ch)
{
using namespace std;
return static_cast<char>(tolower(static_cast<unsigned char>(ch)));
}

template<>
inline char c_toupper(char ch)
{
using namespace std;
return static_cast<char>(toupper(static_cast<unsigned char>(ch)));
}

#ifndef BOOST_XPRESSIVE_NO_WREGEX
template<>
inline wchar_t c_tolower(wchar_t ch)
{
using namespace std;
return towlower(ch);
}

template<>
inline wchar_t c_toupper(wchar_t ch)
{
using namespace std;
return towupper(ch);
}
#endif

} 

struct regex_traits_version_1_tag;

template<typename Char>
struct c_regex_traits
: detail::c_regex_traits_base<Char>
{
typedef Char char_type;
typedef std::basic_string<char_type> string_type;
typedef detail::empty_locale locale_type;
typedef typename detail::char_class_impl<Char>::char_class_type char_class_type;
typedef regex_traits_version_2_tag version_tag;
typedef detail::c_regex_traits_base<Char> base_type;

c_regex_traits(locale_type const &loc = locale_type())
: base_type()
{
this->imbue(loc);
}

bool operator ==(c_regex_traits<char_type> const &) const
{
return true;
}

bool operator !=(c_regex_traits<char_type> const &) const
{
return false;
}

static char_type widen(char ch);

static unsigned char hash(char_type ch)
{
return static_cast<unsigned char>(std::char_traits<Char>::to_int_type(ch));
}

static char_type translate(char_type ch)
{
return ch;
}

static char_type translate_nocase(char_type ch)
{
return detail::c_tolower(ch);
}

static char_type tolower(char_type ch)
{
return detail::c_tolower(ch);
}

static char_type toupper(char_type ch)
{
return detail::c_toupper(ch);
}

string_type fold_case(char_type ch) const
{
BOOST_MPL_ASSERT((is_same<char_type, char>));
char_type ntcs[] = {
detail::c_tolower(ch)
, detail::c_toupper(ch)
, 0
};
if(ntcs[1] == ntcs[0])
ntcs[1] = 0;
return string_type(ntcs);
}

static bool in_range(char_type first, char_type last, char_type ch)
{
return first <= ch && ch <= last;
}

static bool in_range_nocase(char_type first, char_type last, char_type ch)
{
return c_regex_traits::in_range(first, last, ch)
|| c_regex_traits::in_range(first, last, detail::c_tolower(ch))
|| c_regex_traits::in_range(first, last, detail::c_toupper(ch));
}

template<typename FwdIter>
static string_type transform(FwdIter begin, FwdIter end)
{
BOOST_ASSERT(false); 
}

template<typename FwdIter>
static string_type transform_primary(FwdIter begin, FwdIter end)
{
BOOST_ASSERT(false); 
}

template<typename FwdIter>
static string_type lookup_collatename(FwdIter begin, FwdIter end)
{
BOOST_ASSERT(false); 
}

template<typename FwdIter>
static char_class_type lookup_classname(FwdIter begin, FwdIter end, bool icase)
{
return detail::char_class_impl<char_type>::lookup_classname(begin, end, icase);
}

static bool isctype(char_type ch, char_class_type mask)
{
return detail::char_class_impl<char_type>::isctype(ch, mask);
}

static int value(char_type ch, int radix);

locale_type imbue(locale_type loc)
{
this->base_type::imbue(*this);
return loc;
}

static locale_type getloc()
{
locale_type loc;
return loc;
}
};

template<>
inline char c_regex_traits<char>::widen(char ch)
{
return ch;
}

#ifndef BOOST_XPRESSIVE_NO_WREGEX
template<>
inline wchar_t c_regex_traits<wchar_t>::widen(char ch)
{
using namespace std;
return btowc(ch);
}
#endif

template<>
inline unsigned char c_regex_traits<char>::hash(char ch)
{
return static_cast<unsigned char>(ch);
}

#ifndef BOOST_XPRESSIVE_NO_WREGEX
template<>
inline unsigned char c_regex_traits<wchar_t>::hash(wchar_t ch)
{
return static_cast<unsigned char>(ch);
}
#endif

template<>
inline int c_regex_traits<char>::value(char ch, int radix)
{
using namespace std;
BOOST_ASSERT(8 == radix || 10 == radix || 16 == radix);
char begin[2] = { ch, '\0' }, *end = 0;
int val = strtol(begin, &end, radix);
return begin == end ? -1 : val;
}

#ifndef BOOST_XPRESSIVE_NO_WREGEX
template<>
inline int c_regex_traits<wchar_t>::value(wchar_t ch, int radix)
{
using namespace std;
BOOST_ASSERT(8 == radix || 10 == radix || 16 == radix);
wchar_t begin[2] = { ch, L'\0' }, *end = 0;
int val = wcstol(begin, &end, radix);
return begin == end ? -1 : val;
}
#endif

template<>
struct has_fold_case<c_regex_traits<char> >
: mpl::true_
{
};

}}

#endif
