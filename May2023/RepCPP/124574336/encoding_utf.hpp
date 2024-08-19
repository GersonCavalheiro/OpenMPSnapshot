#ifndef BOOST_LOCALE_ENCODING_UTF_HPP_INCLUDED
#define BOOST_LOCALE_ENCODING_UTF_HPP_INCLUDED

#include <boost/locale/utf.hpp>
#include <boost/locale/encoding_errors.hpp>
#include <iterator>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif



namespace boost {
namespace locale {
namespace conv {

template<typename CharOut,typename CharIn>
std::basic_string<CharOut>
utf_to_utf(CharIn const *begin,CharIn const *end,method_type how = default_method)
{
std::basic_string<CharOut> result;
result.reserve(end-begin);
typedef std::back_insert_iterator<std::basic_string<CharOut> > inserter_type;
inserter_type inserter(result);
utf::code_point c;
while(begin!=end) {
c=utf::utf_traits<CharIn>::template decode<CharIn const *>(begin,end);
if(c==utf::illegal || c==utf::incomplete) {
if(how==stop)
throw conversion_error();
}
else {
utf::utf_traits<CharOut>::template encode<inserter_type>(c,inserter);
}
}
return result;
}

template<typename CharOut,typename CharIn>
std::basic_string<CharOut>
utf_to_utf(CharIn const *str,method_type how = default_method)
{
CharIn const *end = str;
while(*end)
end++;
return utf_to_utf<CharOut,CharIn>(str,end,how);
}


template<typename CharOut,typename CharIn>
std::basic_string<CharOut>
utf_to_utf(std::basic_string<CharIn> const &str,method_type how = default_method)
{
return utf_to_utf<CharOut,CharIn>(str.c_str(),str.c_str()+str.size(),how);
}



} 

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif


