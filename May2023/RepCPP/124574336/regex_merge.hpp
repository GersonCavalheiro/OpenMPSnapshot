



#ifndef BOOST_REGEX_V4_REGEX_MERGE_HPP
#define BOOST_REGEX_V4_REGEX_MERGE_HPP


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

template <class OutputIterator, class Iterator, class traits, class charT>
inline OutputIterator regex_merge(OutputIterator out,
Iterator first,
Iterator last,
const basic_regex<charT, traits>& e, 
const charT* fmt, 
match_flag_type flags = match_default)
{
return regex_replace(out, first, last, e, fmt, flags);
}

template <class OutputIterator, class Iterator, class traits, class charT>
inline OutputIterator regex_merge(OutputIterator out,
Iterator first,
Iterator last,
const basic_regex<charT, traits>& e, 
const std::basic_string<charT>& fmt,
match_flag_type flags = match_default)
{
return regex_merge(out, first, last, e, fmt.c_str(), flags);
}

template <class traits, class charT>
inline std::basic_string<charT> regex_merge(const std::basic_string<charT>& s,
const basic_regex<charT, traits>& e, 
const charT* fmt,
match_flag_type flags = match_default)
{
return regex_replace(s, e, fmt, flags);
}

template <class traits, class charT>
inline std::basic_string<charT> regex_merge(const std::basic_string<charT>& s,
const basic_regex<charT, traits>& e, 
const std::basic_string<charT>& fmt,
match_flag_type flags = match_default)
{
return regex_replace(s, e, fmt, flags);
}

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


