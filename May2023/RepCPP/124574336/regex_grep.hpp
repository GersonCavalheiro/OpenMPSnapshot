



#ifndef BOOST_REGEX_V4_REGEX_GREP_HPP
#define BOOST_REGEX_V4_REGEX_GREP_HPP


namespace boost{

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class Predicate, class BidiIterator, class charT, class traits>
inline unsigned int regex_grep(Predicate foo, 
BidiIterator first, 
BidiIterator last, 
const basic_regex<charT, traits>& e, 
match_flag_type flags = match_default)
{
if(e.flags() & regex_constants::failbit)
return false;

typedef typename match_results<BidiIterator>::allocator_type match_allocator_type;

match_results<BidiIterator> m;
BOOST_REGEX_DETAIL_NS::perl_matcher<BidiIterator, match_allocator_type, traits> matcher(first, last, m, e, flags, first);
unsigned int count = 0;
while(matcher.find())
{
++count;
if(0 == foo(m))
return count; 
if(m[0].second == last)
return count; 
if(m.length() == 0)
{
if(m[0].second == last)
return count;
match_results<BidiIterator, match_allocator_type> m2(m);
matcher.setf(match_not_null | match_continuous);
if(matcher.find())
{
++count;
if(0 == foo(m))
return count;
}
else
{
m = m2;
}
matcher.unsetf((match_not_null | match_continuous) & ~flags);
}
}
return count;
}

#ifndef BOOST_NO_FUNCTION_TEMPLATE_ORDERING
template <class Predicate, class charT, class traits>
inline unsigned int regex_grep(Predicate foo, const charT* str, 
const basic_regex<charT, traits>& e, 
match_flag_type flags = match_default)
{
return regex_grep(foo, str, str + traits::length(str), e, flags);
}

template <class Predicate, class ST, class SA, class charT, class traits>
inline unsigned int regex_grep(Predicate foo, const std::basic_string<charT, ST, SA>& s, 
const basic_regex<charT, traits>& e, 
match_flag_type flags = match_default)
{
return regex_grep(foo, s.begin(), s.end(), e, flags);
}
#else  
inline unsigned int regex_grep(bool (*foo)(const cmatch&), const char* str, 
const regex& e, 
match_flag_type flags = match_default)
{
return regex_grep(foo, str, str + regex::traits_type::length(str), e, flags);
}
#ifndef BOOST_NO_WREGEX
inline unsigned int regex_grep(bool (*foo)(const wcmatch&), const wchar_t* str, 
const wregex& e, 
match_flag_type flags = match_default)
{
return regex_grep(foo, str, str + wregex::traits_type::length(str), e, flags);
}
#endif
inline unsigned int regex_grep(bool (*foo)(const match_results<std::string::const_iterator>&), const std::string& s,
const regex& e, 
match_flag_type flags = match_default)
{
return regex_grep(foo, s.begin(), s.end(), e, flags);
}
#if !defined(BOOST_NO_WREGEX)
inline unsigned int regex_grep(bool (*foo)(const match_results<std::basic_string<wchar_t>::const_iterator>&), 
const std::basic_string<wchar_t>& s, 
const wregex& e, 
match_flag_type flags = match_default)
{
return regex_grep(foo, s.begin(), s.end(), e, flags);
}
#endif
#endif

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4103)
#endif
#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_SUFFIX
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

} 

#endif  

