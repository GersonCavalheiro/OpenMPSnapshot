
#ifndef BOOST_XPRESSIVE_TRAITS_CPP_REGEX_TRAITS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_TRAITS_CPP_REGEX_TRAITS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <ios>
#include <string>
#include <locale>
#include <sstream>
#include <climits>
#include <boost/config.hpp>
#include <boost/assert.hpp>
#include <boost/integer.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/utility/literals.hpp>

#if BOOST_WORKAROUND(__GLIBCPP__, != 0)
# define BOOST_XPRESSIVE_BUGGY_CTYPE_FACET
#endif

namespace boost { namespace xpressive
{

namespace detail
{
typedef boost::uint_t<sizeof(std::ctype_base::mask) * CHAR_BIT>::least umask_t;
BOOST_MPL_ASSERT_RELATION(sizeof(std::ctype_base::mask), ==, sizeof(umask_t));

int const umaskex_bits = (14 > (sizeof(umask_t) * CHAR_BIT)) ? 14 : sizeof(umask_t) * CHAR_BIT;

typedef boost::uint_t<umaskex_bits>::fast umaskex_t;
BOOST_MPL_ASSERT_RELATION(sizeof(umask_t), <=, sizeof(umaskex_t));

template<std::ctype_base::mask Mask>
struct mask_cast
{
BOOST_STATIC_CONSTANT(umaskex_t, value = static_cast<umask_t>(Mask));
};

#ifdef __CYGWIN__
template<>
struct mask_cast<std::ctype_base::print>
{
BOOST_MPL_ASSERT_RELATION('\227', ==, std::ctype_base::print);
BOOST_STATIC_CONSTANT(umaskex_t, value = 0227);
};
#endif

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<std::ctype_base::mask Mask>
umaskex_t const mask_cast<Mask>::value;
#endif

#ifndef BOOST_XPRESSIVE_BUGGY_CTYPE_FACET
umaskex_t const highest_bit = static_cast<umaskex_t>(1) << (sizeof(umaskex_t) * CHAR_BIT - 1);

template<umaskex_t In, umaskex_t Out = highest_bit, bool Done = (0 == (Out & In))>
struct unused_mask
{
BOOST_STATIC_ASSERT(1 != Out);
BOOST_STATIC_CONSTANT(umaskex_t, value = (unused_mask<In, (Out >> 1)>::value));
};

template<umaskex_t In, umaskex_t Out>
struct unused_mask<In, Out, true>
{
BOOST_STATIC_CONSTANT(umaskex_t, value = Out);
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<umaskex_t In, umaskex_t Out, bool Done>
umaskex_t const unused_mask<In, Out, Done>::value;
#endif

umaskex_t const std_ctype_alnum = mask_cast<std::ctype_base::alnum>::value;
umaskex_t const std_ctype_alpha = mask_cast<std::ctype_base::alpha>::value;
umaskex_t const std_ctype_cntrl = mask_cast<std::ctype_base::cntrl>::value;
umaskex_t const std_ctype_digit = mask_cast<std::ctype_base::digit>::value;
umaskex_t const std_ctype_graph = mask_cast<std::ctype_base::graph>::value;
umaskex_t const std_ctype_lower = mask_cast<std::ctype_base::lower>::value;
umaskex_t const std_ctype_print = mask_cast<std::ctype_base::print>::value;
umaskex_t const std_ctype_punct = mask_cast<std::ctype_base::punct>::value;
umaskex_t const std_ctype_space = mask_cast<std::ctype_base::space>::value;
umaskex_t const std_ctype_upper = mask_cast<std::ctype_base::upper>::value;
umaskex_t const std_ctype_xdigit = mask_cast<std::ctype_base::xdigit>::value;

#if defined(__GLIBCXX__)
umaskex_t const std_ctype_reserved = 0x8000;
#elif defined(_CPPLIB_VER) && defined(BOOST_WINDOWS)
umaskex_t const std_ctype_reserved = 0x8200;
#elif defined(_LIBCPP_VERSION)
umaskex_t const std_ctype_reserved = 0x8000;
#else
umaskex_t const std_ctype_reserved = 0;
#endif

umaskex_t const all_ctype_masks = std_ctype_reserved
| std_ctype_alnum | std_ctype_alpha | std_ctype_cntrl | std_ctype_digit
| std_ctype_graph | std_ctype_lower | std_ctype_print | std_ctype_punct
| std_ctype_space | std_ctype_upper | std_ctype_xdigit;

umaskex_t const non_std_ctype_underscore = unused_mask<all_ctype_masks>::value;

umaskex_t const non_std_ctype_blank = unused_mask<all_ctype_masks | non_std_ctype_underscore>::value;

umaskex_t const non_std_ctype_newline = unused_mask<all_ctype_masks | non_std_ctype_underscore | non_std_ctype_blank>::value;

#else
umaskex_t const std_ctype_alnum = 1 << 0;
umaskex_t const std_ctype_alpha = 1 << 1;
umaskex_t const std_ctype_cntrl = 1 << 2;
umaskex_t const std_ctype_digit = 1 << 3;
umaskex_t const std_ctype_graph = 1 << 4;
umaskex_t const std_ctype_lower = 1 << 5;
umaskex_t const std_ctype_print = 1 << 6;
umaskex_t const std_ctype_punct = 1 << 7;
umaskex_t const std_ctype_space = 1 << 8;
umaskex_t const std_ctype_upper = 1 << 9;
umaskex_t const std_ctype_xdigit = 1 << 10;
umaskex_t const non_std_ctype_underscore = 1 << 11;
umaskex_t const non_std_ctype_blank = 1 << 12;
umaskex_t const non_std_ctype_newline = 1 << 13;

static umaskex_t const std_masks[] =
{
mask_cast<std::ctype_base::alnum>::value
, mask_cast<std::ctype_base::alpha>::value
, mask_cast<std::ctype_base::cntrl>::value
, mask_cast<std::ctype_base::digit>::value
, mask_cast<std::ctype_base::graph>::value
, mask_cast<std::ctype_base::lower>::value
, mask_cast<std::ctype_base::print>::value
, mask_cast<std::ctype_base::punct>::value
, mask_cast<std::ctype_base::space>::value
, mask_cast<std::ctype_base::upper>::value
, mask_cast<std::ctype_base::xdigit>::value
};

inline int mylog2(umaskex_t i)
{
return "\0\0\1\0\2\0\0\0\3"[i & 0xf]
+ "\0\4\5\0\6\0\0\0\7"[(i & 0xf0) >> 04]
+ "\0\10\11\0\12\0\0\0\13"[(i & 0xf00) >> 010];
}
#endif

umaskex_t const non_std_ctype_masks = non_std_ctype_underscore | non_std_ctype_blank | non_std_ctype_newline;

template<typename Char, std::size_t SizeOfChar = sizeof(Char)>
struct cpp_regex_traits_base
{
protected:
void imbue(std::locale const &)
{
}

static bool is(std::ctype<Char> const &ct, Char ch, umaskex_t mask)
{
#ifndef BOOST_XPRESSIVE_BUGGY_CTYPE_FACET

if(ct.is((std::ctype_base::mask)(umask_t)mask, ch))
{
return true;
}

#if defined(__CYGWIN__) || defined(__MINGW32_VERSION)
if (std::ctype_base::xdigit == ((std::ctype_base::mask)(umask_t)mask & std::ctype_base::xdigit))
{
typename std::char_traits<Char>::int_type i = std::char_traits<Char>::to_int_type(ch);
if(UCHAR_MAX >= i && std::isxdigit(static_cast<int>(i)))
return true;
}
#endif

#else

umaskex_t tmp = mask & ~non_std_ctype_masks;
for(umaskex_t i; 0 != (i = (tmp & (~tmp+1))); tmp &= ~i)
{
std::ctype_base::mask m = (std::ctype_base::mask)(umask_t)std_masks[mylog2(i)];
if(ct.is(m, ch))
{
return true;
}
}

#endif

return ((mask & non_std_ctype_blank) && cpp_regex_traits_base::is_blank(ch))
|| ((mask & non_std_ctype_underscore) && cpp_regex_traits_base::is_underscore(ch))
|| ((mask & non_std_ctype_newline) && cpp_regex_traits_base::is_newline(ch));
}

private:
static bool is_blank(Char ch)
{
BOOST_MPL_ASSERT_RELATION('\t', ==, L'\t');
BOOST_MPL_ASSERT_RELATION(' ', ==, L' ');
return L' ' == ch || L'\t' == ch;
}

static bool is_underscore(Char ch)
{
BOOST_MPL_ASSERT_RELATION('_', ==, L'_');
return L'_' == ch;
}

static bool is_newline(Char ch)
{
BOOST_MPL_ASSERT_RELATION('\r', ==, L'\r');
BOOST_MPL_ASSERT_RELATION('\n', ==, L'\n');
BOOST_MPL_ASSERT_RELATION('\f', ==, L'\f');
return L'\r' == ch || L'\n' == ch || L'\f' == ch
|| (1 < SizeOfChar && (0x2028u == ch || 0x2029u == ch || 0x85u == ch));
}
};

#ifndef BOOST_XPRESSIVE_BUGGY_CTYPE_FACET

template<typename Char>
struct cpp_regex_traits_base<Char, 1>
{
protected:
void imbue(std::locale const &loc)
{
int i = 0;
Char allchars[UCHAR_MAX + 1];
for(i = 0; i <= static_cast<int>(UCHAR_MAX); ++i)
{
allchars[i] = static_cast<Char>(i);
}

std::ctype<Char> const &ct = BOOST_USE_FACET(std::ctype<Char>, loc);
std::ctype_base::mask tmp[UCHAR_MAX + 1];
ct.is(allchars, allchars + UCHAR_MAX + 1, tmp);
for(i = 0; i <= static_cast<int>(UCHAR_MAX); ++i)
{
this->masks_[i] = static_cast<umask_t>(tmp[i]);
BOOST_ASSERT(0 == (this->masks_[i] & non_std_ctype_masks));
}

this->masks_[static_cast<unsigned char>('_')] |= non_std_ctype_underscore;
this->masks_[static_cast<unsigned char>(' ')] |= non_std_ctype_blank;
this->masks_[static_cast<unsigned char>('\t')] |= non_std_ctype_blank;
this->masks_[static_cast<unsigned char>('\n')] |= non_std_ctype_newline;
this->masks_[static_cast<unsigned char>('\r')] |= non_std_ctype_newline;
this->masks_[static_cast<unsigned char>('\f')] |= non_std_ctype_newline;
}

bool is(std::ctype<Char> const &, Char ch, umaskex_t mask) const
{
return 0 != (this->masks_[static_cast<unsigned char>(ch)] & mask);
}

private:
umaskex_t masks_[UCHAR_MAX + 1];
};

#endif

} 


template<typename Char>
struct cpp_regex_traits
: detail::cpp_regex_traits_base<Char>
{
typedef Char char_type;
typedef std::basic_string<char_type> string_type;
typedef std::locale locale_type;
typedef detail::umaskex_t char_class_type;
typedef regex_traits_version_2_tag version_tag;
typedef detail::cpp_regex_traits_base<Char> base_type;

cpp_regex_traits(locale_type const &loc = locale_type())
: base_type()
, loc_()
{
this->imbue(loc);
}

bool operator ==(cpp_regex_traits<char_type> const &that) const
{
return this->loc_ == that.loc_;
}

bool operator !=(cpp_regex_traits<char_type> const &that) const
{
return this->loc_ != that.loc_;
}

char_type widen(char ch) const
{
return this->ctype_->widen(ch);
}

static unsigned char hash(char_type ch)
{
return static_cast<unsigned char>(std::char_traits<Char>::to_int_type(ch));
}

static char_type translate(char_type ch)
{
return ch;
}

char_type translate_nocase(char_type ch) const
{
return this->ctype_->tolower(ch);
}

char_type tolower(char_type ch) const
{
return this->ctype_->tolower(ch);
}

char_type toupper(char_type ch) const
{
return this->ctype_->toupper(ch);
}

string_type fold_case(char_type ch) const
{
BOOST_MPL_ASSERT((is_same<char_type, char>));
char_type ntcs[] = {
this->ctype_->tolower(ch)
, this->ctype_->toupper(ch)
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

bool in_range_nocase(char_type first, char_type last, char_type ch) const
{
return this->in_range(first, last, ch)
|| this->in_range(first, last, this->ctype_->toupper(ch))
|| this->in_range(first, last, this->ctype_->tolower(ch));
}


template<typename FwdIter>
string_type transform(FwdIter, FwdIter) const
{

BOOST_ASSERT(false);
return string_type();
}

template<typename FwdIter>
string_type transform_primary(FwdIter, FwdIter ) const
{
BOOST_ASSERT(false); 
return string_type();
}

template<typename FwdIter>
string_type lookup_collatename(FwdIter, FwdIter) const
{
BOOST_ASSERT(false); 
return string_type();
}

template<typename FwdIter>
char_class_type lookup_classname(FwdIter begin, FwdIter end, bool icase) const
{
static detail::umaskex_t const icase_masks =
detail::std_ctype_lower | detail::std_ctype_upper;

BOOST_ASSERT(begin != end);
char_class_type char_class = this->lookup_classname_impl_(begin, end);
if(0 == char_class)
{
string_type classname(begin, end);
for(typename string_type::size_type i = 0, len = classname.size(); i < len; ++i)
{
classname[i] = this->translate_nocase(classname[i]);
}
char_class = this->lookup_classname_impl_(classname.begin(), classname.end());
}
if(icase && 0 != (char_class & icase_masks))
{
char_class |= icase_masks;
}
return char_class;
}

bool isctype(char_type ch, char_class_type mask) const
{
return this->base_type::is(*this->ctype_, ch, mask);
}

int value(char_type ch, int radix) const
{
BOOST_ASSERT(8 == radix || 10 == radix || 16 == radix);
int val = -1;
std::basic_stringstream<char_type> str;
str.imbue(this->getloc());
str << (8 == radix ? std::oct : (16 == radix ? std::hex : std::dec));
str.put(ch);
str >> val;
return str.fail() ? -1 : val;
}

locale_type imbue(locale_type loc)
{
locale_type old_loc = this->loc_;
this->loc_ = loc;
this->ctype_ = &BOOST_USE_FACET(std::ctype<char_type>, this->loc_);
this->base_type::imbue(this->loc_);
return old_loc;
}

locale_type getloc() const
{
return this->loc_;
}

private:

struct char_class_pair
{
char_type const *class_name_;
char_class_type class_type_;
};

static char_class_pair const &char_class(std::size_t j)
{
static BOOST_CONSTEXPR_OR_CONST char_class_pair s_char_class_map[] =
{
{ BOOST_XPR_CSTR_(char_type, "alnum"),  detail::std_ctype_alnum }
, { BOOST_XPR_CSTR_(char_type, "alpha"),  detail::std_ctype_alpha }
, { BOOST_XPR_CSTR_(char_type, "blank"),  detail::non_std_ctype_blank }
, { BOOST_XPR_CSTR_(char_type, "cntrl"),  detail::std_ctype_cntrl }
, { BOOST_XPR_CSTR_(char_type, "d"),      detail::std_ctype_digit }
, { BOOST_XPR_CSTR_(char_type, "digit"),  detail::std_ctype_digit }
, { BOOST_XPR_CSTR_(char_type, "graph"),  detail::std_ctype_graph }
, { BOOST_XPR_CSTR_(char_type, "lower"),  detail::std_ctype_lower }
, { BOOST_XPR_CSTR_(char_type, "newline"),detail::non_std_ctype_newline }
, { BOOST_XPR_CSTR_(char_type, "print"),  detail::std_ctype_print }
, { BOOST_XPR_CSTR_(char_type, "punct"),  detail::std_ctype_punct }
, { BOOST_XPR_CSTR_(char_type, "s"),      detail::std_ctype_space }
, { BOOST_XPR_CSTR_(char_type, "space"),  detail::std_ctype_space }
, { BOOST_XPR_CSTR_(char_type, "upper"),  detail::std_ctype_upper }
, { BOOST_XPR_CSTR_(char_type, "w"),      detail::std_ctype_alnum | detail::non_std_ctype_underscore }
, { BOOST_XPR_CSTR_(char_type, "xdigit"), detail::std_ctype_xdigit }
, { 0, 0 }
};
return s_char_class_map[j];
}

template<typename FwdIter>
static char_class_type lookup_classname_impl_(FwdIter begin, FwdIter end)
{
typedef cpp_regex_traits<Char> this_t;
for(std::size_t j = 0; 0 != this_t::char_class(j).class_name_; ++j)
{
if(this_t::compare_(this_t::char_class(j).class_name_, begin, end))
{
return this_t::char_class(j).class_type_;
}
}
return 0;
}

template<typename FwdIter>
static bool compare_(char_type const *name, FwdIter begin, FwdIter end)
{
for(; *name && begin != end; ++name, ++begin)
{
if(*name != *begin)
{
return false;
}
}
return !*name && begin == end;
}

locale_type loc_;
std::ctype<char_type> const *ctype_;
};

template<>
inline unsigned char cpp_regex_traits<unsigned char>::hash(unsigned char ch)
{
return ch;
}

template<>
inline unsigned char cpp_regex_traits<char>::hash(char ch)
{
return static_cast<unsigned char>(ch);
}

template<>
inline unsigned char cpp_regex_traits<signed char>::hash(signed char ch)
{
return static_cast<unsigned char>(ch);
}

#ifndef BOOST_XPRESSIVE_NO_WREGEX
template<>
inline unsigned char cpp_regex_traits<wchar_t>::hash(wchar_t ch)
{
return static_cast<unsigned char>(ch);
}
#endif

template<>
struct has_fold_case<cpp_regex_traits<char> >
: mpl::true_
{
};


}}

#endif
