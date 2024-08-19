


#define BOOST_REGEX_SOURCE

#include <boost/regex/config.hpp>
#ifdef BOOST_HAS_ICU
#define BOOST_REGEX_ICU_INSTANTIATE
#include <boost/regex/icu.hpp>

#ifdef BOOST_INTEL
#pragma warning(disable:981 2259 383)
#endif

namespace boost{

namespace BOOST_REGEX_DETAIL_NS{

icu_regex_traits_implementation::string_type icu_regex_traits_implementation::do_transform(const char_type* p1, const char_type* p2, const U_NAMESPACE_QUALIFIER Collator* pcoll) const
{
typedef u32_to_u16_iterator<const char_type*, ::UChar> itt;
itt i(p1), j(p2);
#ifndef BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
std::vector< ::UChar> t(i, j);
#else
std::vector< ::UChar> t;
while(i != j)
t.push_back(*i++);
#endif
::uint8_t result[100];
::int32_t len;
if(t.size())
len = pcoll->getSortKey(&*t.begin(), static_cast< ::int32_t>(t.size()), result, sizeof(result));
else
len = pcoll->getSortKey(static_cast<UChar const*>(0), static_cast< ::int32_t>(0), result, sizeof(result));
if(std::size_t(len) > sizeof(result))
{
scoped_array< ::uint8_t> presult(new ::uint8_t[len+1]);
if(t.size())
len = pcoll->getSortKey(&*t.begin(), static_cast< ::int32_t>(t.size()), presult.get(), len+1);
else
len = pcoll->getSortKey(static_cast<UChar const*>(0), static_cast< ::int32_t>(0), presult.get(), len+1);
if((0 == presult[len-1]) && (len > 1))
--len;
#ifndef BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
return string_type(presult.get(), presult.get()+len);
#else
string_type sresult;
::uint8_t const* ia = presult.get();
::uint8_t const* ib = presult.get()+len;
while(ia != ib)
sresult.push_back(*ia++);
return sresult;
#endif
}
if((0 == result[len-1]) && (len > 1))
--len;
#ifndef BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
return string_type(result, result+len);
#else
string_type sresult;
::uint8_t const* ia = result;
::uint8_t const* ib = result+len;
while(ia != ib)
sresult.push_back(*ia++);
return sresult;
#endif
}

}

icu_regex_traits::size_type icu_regex_traits::length(const char_type* p)
{
size_type result = 0;
while(*p)
{
++p;
++result;
}
return result;
}

const icu_regex_traits::char_class_type icu_regex_traits::mask_blank = icu_regex_traits::char_class_type(1) << offset_blank;
const icu_regex_traits::char_class_type icu_regex_traits::mask_space = icu_regex_traits::char_class_type(1) << offset_space;
const icu_regex_traits::char_class_type icu_regex_traits::mask_xdigit = icu_regex_traits::char_class_type(1) << offset_xdigit;
const icu_regex_traits::char_class_type icu_regex_traits::mask_underscore = icu_regex_traits::char_class_type(1) << offset_underscore;
const icu_regex_traits::char_class_type icu_regex_traits::mask_unicode = icu_regex_traits::char_class_type(1) << offset_unicode;
const icu_regex_traits::char_class_type icu_regex_traits::mask_any = icu_regex_traits::char_class_type(1) << offset_any;
const icu_regex_traits::char_class_type icu_regex_traits::mask_ascii = icu_regex_traits::char_class_type(1) << offset_ascii;
const icu_regex_traits::char_class_type icu_regex_traits::mask_horizontal = icu_regex_traits::char_class_type(1) << offset_horizontal;
const icu_regex_traits::char_class_type icu_regex_traits::mask_vertical = icu_regex_traits::char_class_type(1) << offset_vertical;

icu_regex_traits::char_class_type icu_regex_traits::lookup_icu_mask(const ::UChar32* p1, const ::UChar32* p2)
{
static const ::UChar32 prop_name_table[] = {
'a', 'n', 'y', 
'a', 's', 'c', 'i', 'i', 
'a', 's', 's', 'i', 'g', 'n', 'e', 'd', 
'c', '*', 
'c', 'c', 
'c', 'f', 
'c', 'l', 'o', 's', 'e', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'c', 'n', 
'c', 'o', 
'c', 'o', 'n', 'n', 'e', 'c', 't', 'o', 'r', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'c', 'o', 'n', 't', 'r', 'o', 'l', 
'c', 's', 
'c', 'u', 'r', 'r', 'e', 'n', 'c', 'y', 's', 'y', 'm', 'b', 'o', 'l', 
'd', 'a', 's', 'h', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'd', 'e', 'c', 'i', 'm', 'a', 'l', 'd', 'i', 'g', 'i', 't', 'n', 'u', 'm', 'b', 'e', 'r', 
'e', 'n', 'c', 'l', 'o', 's', 'i', 'n', 'g', 'm', 'a', 'r', 'k', 
'f', 'i', 'n', 'a', 'l', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'f', 'o', 'r', 'm', 'a', 't', 
'i', 'n', 'i', 't', 'i', 'a', 'l', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'l', '*', 
'l', 'e', 't', 't', 'e', 'r', 
'l', 'e', 't', 't', 'e', 'r', 'n', 'u', 'm', 'b', 'e', 'r', 
'l', 'i', 'n', 'e', 's', 'e', 'p', 'a', 'r', 'a', 't', 'o', 'r', 
'l', 'l', 
'l', 'm', 
'l', 'o', 
'l', 'o', 'w', 'e', 'r', 'c', 'a', 's', 'e', 'l', 'e', 't', 't', 'e', 'r', 
'l', 't', 
'l', 'u', 
'm', '*', 
'm', 'a', 'r', 'k', 
'm', 'a', 't', 'h', 's', 'y', 'm', 'b', 'o', 'l', 
'm', 'c', 
'm', 'e', 
'm', 'n', 
'm', 'o', 'd', 'i', 'f', 'i', 'e', 'r', 'l', 'e', 't', 't', 'e', 'r', 
'm', 'o', 'd', 'i', 'f', 'i', 'e', 'r', 's', 'y', 'm', 'b', 'o', 'l', 
'n', '*', 
'n', 'd', 
'n', 'l', 
'n', 'o', 
'n', 'o', 'n', 's', 'p', 'a', 'c', 'i', 'n', 'g', 'm', 'a', 'r', 'k', 
'n', 'o', 't', 'a', 's', 's', 'i', 'g', 'n', 'e', 'd', 
'n', 'u', 'm', 'b', 'e', 'r', 
'o', 'p', 'e', 'n', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'o', 't', 'h', 'e', 'r', 
'o', 't', 'h', 'e', 'r', 'l', 'e', 't', 't', 'e', 'r', 
'o', 't', 'h', 'e', 'r', 'n', 'u', 'm', 'b', 'e', 'r', 
'o', 't', 'h', 'e', 'r', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
'o', 't', 'h', 'e', 'r', 's', 'y', 'm', 'b', 'o', 'l', 
'p', '*', 
'p', 'a', 'r', 'a', 'g', 'r', 'a', 'p', 'h', 's', 'e', 'p', 'a', 'r', 'a', 't', 'o', 'r', 
'p', 'c', 
'p', 'd', 
'p', 'e', 
'p', 'f', 
'p', 'i', 
'p', 'o', 
'p', 'r', 'i', 'v', 'a', 't', 'e', 'u', 's', 'e', 
'p', 's', 
'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n', 
's', '*', 
's', 'c', 
's', 'e', 'p', 'a', 'r', 'a', 't', 'o', 'r', 
's', 'k', 
's', 'm', 
's', 'o', 
's', 'p', 'a', 'c', 'e', 's', 'e', 'p', 'a', 'r', 'a', 't', 'o', 'r', 
's', 'p', 'a', 'c', 'i', 'n', 'g', 'c', 'o', 'm', 'b', 'i', 'n', 'i', 'n', 'g', 'm', 'a', 'r', 'k', 
's', 'u', 'r', 'r', 'o', 'g', 'a', 't', 'e', 
's', 'y', 'm', 'b', 'o', 'l', 
't', 'i', 't', 'l', 'e', 'c', 'a', 's', 'e', 
't', 'i', 't', 'l', 'e', 'c', 'a', 's', 'e', 'l', 'e', 't', 't', 'e', 'r', 
'u', 'p', 'p', 'e', 'r', 'c', 'a', 's', 'e', 'l', 'e', 't', 't', 'e', 'r', 
'z', '*', 
'z', 'l', 
'z', 'p', 
'z', 's', 
};

static const BOOST_REGEX_DETAIL_NS::character_pointer_range< ::UChar32> range_data[] = {
{ prop_name_table+0, prop_name_table+3, }, 
{ prop_name_table+3, prop_name_table+8, }, 
{ prop_name_table+8, prop_name_table+16, }, 
{ prop_name_table+16, prop_name_table+18, }, 
{ prop_name_table+18, prop_name_table+20, }, 
{ prop_name_table+20, prop_name_table+22, }, 
{ prop_name_table+22, prop_name_table+38, }, 
{ prop_name_table+38, prop_name_table+40, }, 
{ prop_name_table+40, prop_name_table+42, }, 
{ prop_name_table+42, prop_name_table+62, }, 
{ prop_name_table+62, prop_name_table+69, }, 
{ prop_name_table+69, prop_name_table+71, }, 
{ prop_name_table+71, prop_name_table+85, }, 
{ prop_name_table+85, prop_name_table+100, }, 
{ prop_name_table+100, prop_name_table+118, }, 
{ prop_name_table+118, prop_name_table+131, }, 
{ prop_name_table+131, prop_name_table+147, }, 
{ prop_name_table+147, prop_name_table+153, }, 
{ prop_name_table+153, prop_name_table+171, }, 
{ prop_name_table+171, prop_name_table+173, }, 
{ prop_name_table+173, prop_name_table+179, }, 
{ prop_name_table+179, prop_name_table+191, }, 
{ prop_name_table+191, prop_name_table+204, }, 
{ prop_name_table+204, prop_name_table+206, }, 
{ prop_name_table+206, prop_name_table+208, }, 
{ prop_name_table+208, prop_name_table+210, }, 
{ prop_name_table+210, prop_name_table+225, }, 
{ prop_name_table+225, prop_name_table+227, }, 
{ prop_name_table+227, prop_name_table+229, }, 
{ prop_name_table+229, prop_name_table+231, }, 
{ prop_name_table+231, prop_name_table+235, }, 
{ prop_name_table+235, prop_name_table+245, }, 
{ prop_name_table+245, prop_name_table+247, }, 
{ prop_name_table+247, prop_name_table+249, }, 
{ prop_name_table+249, prop_name_table+251, }, 
{ prop_name_table+251, prop_name_table+265, }, 
{ prop_name_table+265, prop_name_table+279, }, 
{ prop_name_table+279, prop_name_table+281, }, 
{ prop_name_table+281, prop_name_table+283, }, 
{ prop_name_table+283, prop_name_table+285, }, 
{ prop_name_table+285, prop_name_table+287, }, 
{ prop_name_table+287, prop_name_table+301, }, 
{ prop_name_table+301, prop_name_table+312, }, 
{ prop_name_table+312, prop_name_table+318, }, 
{ prop_name_table+318, prop_name_table+333, }, 
{ prop_name_table+333, prop_name_table+338, }, 
{ prop_name_table+338, prop_name_table+349, }, 
{ prop_name_table+349, prop_name_table+360, }, 
{ prop_name_table+360, prop_name_table+376, }, 
{ prop_name_table+376, prop_name_table+387, }, 
{ prop_name_table+387, prop_name_table+389, }, 
{ prop_name_table+389, prop_name_table+407, }, 
{ prop_name_table+407, prop_name_table+409, }, 
{ prop_name_table+409, prop_name_table+411, }, 
{ prop_name_table+411, prop_name_table+413, }, 
{ prop_name_table+413, prop_name_table+415, }, 
{ prop_name_table+415, prop_name_table+417, }, 
{ prop_name_table+417, prop_name_table+419, }, 
{ prop_name_table+419, prop_name_table+429, }, 
{ prop_name_table+429, prop_name_table+431, }, 
{ prop_name_table+431, prop_name_table+442, }, 
{ prop_name_table+442, prop_name_table+444, }, 
{ prop_name_table+444, prop_name_table+446, }, 
{ prop_name_table+446, prop_name_table+455, }, 
{ prop_name_table+455, prop_name_table+457, }, 
{ prop_name_table+457, prop_name_table+459, }, 
{ prop_name_table+459, prop_name_table+461, }, 
{ prop_name_table+461, prop_name_table+475, }, 
{ prop_name_table+475, prop_name_table+495, }, 
{ prop_name_table+495, prop_name_table+504, }, 
{ prop_name_table+504, prop_name_table+510, }, 
{ prop_name_table+510, prop_name_table+519, }, 
{ prop_name_table+519, prop_name_table+534, }, 
{ prop_name_table+534, prop_name_table+549, }, 
{ prop_name_table+549, prop_name_table+551, }, 
{ prop_name_table+551, prop_name_table+553, }, 
{ prop_name_table+553, prop_name_table+555, }, 
{ prop_name_table+555, prop_name_table+557, }, 
};

static const icu_regex_traits::char_class_type icu_class_map[] = {
icu_regex_traits::mask_any, 
icu_regex_traits::mask_ascii, 
(0x3FFFFFFFu) & ~(U_GC_CN_MASK), 
U_GC_C_MASK, 
U_GC_CC_MASK, 
U_GC_CF_MASK, 
U_GC_PE_MASK, 
U_GC_CN_MASK, 
U_GC_CO_MASK, 
U_GC_PC_MASK, 
U_GC_CC_MASK, 
U_GC_CS_MASK, 
U_GC_SC_MASK, 
U_GC_PD_MASK, 
U_GC_ND_MASK, 
U_GC_ME_MASK, 
U_GC_PF_MASK, 
U_GC_CF_MASK, 
U_GC_PI_MASK, 
U_GC_L_MASK, 
U_GC_L_MASK, 
U_GC_NL_MASK, 
U_GC_ZL_MASK, 
U_GC_LL_MASK, 
U_GC_LM_MASK, 
U_GC_LO_MASK, 
U_GC_LL_MASK, 
U_GC_LT_MASK, 
U_GC_LU_MASK, 
U_GC_M_MASK, 
U_GC_M_MASK, 
U_GC_SM_MASK, 
U_GC_MC_MASK, 
U_GC_ME_MASK, 
U_GC_MN_MASK, 
U_GC_LM_MASK, 
U_GC_SK_MASK, 
U_GC_N_MASK, 
U_GC_ND_MASK, 
U_GC_NL_MASK, 
U_GC_NO_MASK, 
U_GC_MN_MASK, 
U_GC_CN_MASK, 
U_GC_N_MASK, 
U_GC_PS_MASK, 
U_GC_C_MASK, 
U_GC_LO_MASK, 
U_GC_NO_MASK, 
U_GC_PO_MASK, 
U_GC_SO_MASK, 
U_GC_P_MASK, 
U_GC_ZP_MASK, 
U_GC_PC_MASK, 
U_GC_PD_MASK, 
U_GC_PE_MASK, 
U_GC_PF_MASK, 
U_GC_PI_MASK, 
U_GC_PO_MASK, 
U_GC_CO_MASK, 
U_GC_PS_MASK, 
U_GC_P_MASK, 
U_GC_S_MASK, 
U_GC_SC_MASK, 
U_GC_Z_MASK, 
U_GC_SK_MASK, 
U_GC_SM_MASK, 
U_GC_SO_MASK, 
U_GC_ZS_MASK, 
U_GC_MC_MASK, 
U_GC_CS_MASK, 
U_GC_S_MASK, 
U_GC_LT_MASK, 
U_GC_LT_MASK, 
U_GC_LU_MASK, 
U_GC_Z_MASK, 
U_GC_ZL_MASK, 
U_GC_ZP_MASK, 
U_GC_ZS_MASK, 
};


const BOOST_REGEX_DETAIL_NS::character_pointer_range< ::UChar32>* ranges_begin = range_data;
const BOOST_REGEX_DETAIL_NS::character_pointer_range< ::UChar32>* ranges_end = range_data + (sizeof(range_data)/sizeof(range_data[0]));

BOOST_REGEX_DETAIL_NS::character_pointer_range< ::UChar32> t = { p1, p2, };
const BOOST_REGEX_DETAIL_NS::character_pointer_range< ::UChar32>* p = std::lower_bound(ranges_begin, ranges_end, t);
if((p != ranges_end) && (t == *p))
return icu_class_map[p - ranges_begin];
return 0;
}

icu_regex_traits::char_class_type icu_regex_traits::lookup_classname(const char_type* p1, const char_type* p2) const
{
static const char_class_type masks[] = 
{
0,
U_GC_L_MASK | U_GC_ND_MASK, 
U_GC_L_MASK,
mask_blank,
U_GC_CC_MASK | U_GC_CF_MASK | U_GC_ZL_MASK | U_GC_ZP_MASK,
U_GC_ND_MASK,
U_GC_ND_MASK,
(0x3FFFFFFFu) & ~(U_GC_CC_MASK | U_GC_CF_MASK | U_GC_CS_MASK | U_GC_CN_MASK | U_GC_Z_MASK),
mask_horizontal,
U_GC_LL_MASK,
U_GC_LL_MASK,
~(U_GC_C_MASK),
U_GC_P_MASK,
char_class_type(U_GC_Z_MASK) | mask_space,
char_class_type(U_GC_Z_MASK) | mask_space,
U_GC_LU_MASK,
mask_unicode,
U_GC_LU_MASK,
mask_vertical,
char_class_type(U_GC_L_MASK | U_GC_ND_MASK | U_GC_MN_MASK) | mask_underscore, 
char_class_type(U_GC_L_MASK | U_GC_ND_MASK | U_GC_MN_MASK) | mask_underscore, 
char_class_type(U_GC_ND_MASK) | mask_xdigit,
};

int idx = ::boost::BOOST_REGEX_DETAIL_NS::get_default_class_id(p1, p2);
if(idx >= 0)
return masks[idx+1];
char_class_type result = lookup_icu_mask(p1, p2);
if(result != 0)
return result;

if(idx < 0)
{
string_type s(p1, p2);
string_type::size_type i = 0;
while(i < s.size())
{
s[i] = static_cast<char>((::u_tolower)(s[i]));
if(::u_isspace(s[i]) || (s[i] == '-') || (s[i] == '_'))
s.erase(s.begin()+i, s.begin()+i+1);
else
{
s[i] = static_cast<char>((::u_tolower)(s[i]));
++i;
}
}
if(s.size())
idx = ::boost::BOOST_REGEX_DETAIL_NS::get_default_class_id(&*s.begin(), &*s.begin() + s.size());
if(idx >= 0)
return masks[idx+1];
if(s.size())
result = lookup_icu_mask(&*s.begin(), &*s.begin() + s.size());
if(result != 0)
return result;
}
BOOST_ASSERT(std::size_t(idx+1) < sizeof(masks) / sizeof(masks[0]));
return masks[idx+1];
}

icu_regex_traits::string_type icu_regex_traits::lookup_collatename(const char_type* p1, const char_type* p2) const
{
string_type result;
#ifdef BOOST_NO_CXX98_BINDERS
if(std::find_if(p1, p2, std::bind(std::greater< ::UChar32>(), std::placeholders::_1, 0x7f)) == p2)
#else
if(std::find_if(p1, p2, std::bind2nd(std::greater< ::UChar32>(), 0x7f)) == p2)
#endif
{
#ifndef BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
std::string s(p1, p2);
#else
std::string s;
const char_type* p3 = p1;
while(p3 != p2)
s.append(1, *p3++);
#endif
UErrorCode err = U_ZERO_ERROR;
UChar32 c = ::u_charFromName(U_UNICODE_CHAR_NAME, s.c_str(), &err);
if(U_SUCCESS(err))
{
result.push_back(c);
return result;
}
err = U_ZERO_ERROR;
c = ::u_charFromName(U_EXTENDED_CHAR_NAME, s.c_str(), &err);
if(U_SUCCESS(err))
{
result.push_back(c);
return result;
}
s = ::boost::BOOST_REGEX_DETAIL_NS::lookup_default_collate_name(s);
#ifndef BOOST_NO_TEMPLATED_ITERATOR_CONSTRUCTORS
result.assign(s.begin(), s.end());
#else
result.clear();
std::string::const_iterator si, sj;
si = s.begin();
sj = s.end();
while(si != sj)
result.push_back(*si++);
#endif
}
if(result.empty() && (p2-p1 == 1))
result.push_back(*p1);
return result;
}

bool icu_regex_traits::isctype(char_type c, char_class_type f) const
{
char_class_type m = char_class_type(static_cast<char_class_type>(1) << u_charType(c));
if((m & f) != 0) 
return true;
if(((f & mask_blank) != 0) && u_isblank(c))
return true;
if(((f & mask_space) != 0) && u_isspace(c))
return true;
if(((f & mask_xdigit) != 0) && (u_digit(c, 16) >= 0))
return true;
if(((f & mask_unicode) != 0) && (c >= 0x100))
return true;
if(((f & mask_underscore) != 0) && (c == '_'))
return true;
if(((f & mask_any) != 0) && (c <= 0x10FFFF))
return true;
if(((f & mask_ascii) != 0) && (c <= 0x7F))
return true;
if(((f & mask_vertical) != 0) && (::boost::BOOST_REGEX_DETAIL_NS::is_separator(c) || (c == static_cast<char_type>('\v')) || (m == U_GC_ZL_MASK) || (m == U_GC_ZP_MASK)))
return true;
if(((f & mask_horizontal) != 0) && !::boost::BOOST_REGEX_DETAIL_NS::is_separator(c) && u_isspace(c) && (c != static_cast<char_type>('\v')))
return true;
return false;
}

}

#endif 
