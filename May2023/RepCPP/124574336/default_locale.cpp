
#define BOOST_LOCALE_SOURCE
#include <boost/locale/util.hpp>
#include <boost/config.hpp>
#include <stdlib.h>

#ifdef BOOST_MSVC
#  pragma warning(disable : 4996)
#endif

#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define BOOST_LOCALE_USE_WIN32_API
#endif

namespace boost {
namespace locale {
namespace util {
std::string get_system_locale(bool use_utf8)
{
char const *lang = 0;
if(!lang || !*lang)
lang = getenv("LC_CTYPE");
if(!lang || !*lang)
lang = getenv("LC_ALL");
if(!lang || !*lang)
lang = getenv("LANG");
#ifndef BOOST_LOCALE_USE_WIN32_API
(void)use_utf8; 
if(!lang || !*lang)
lang = "C";
return lang;
#else
if(lang && *lang) {
return lang;
}
char buf[10];
if(GetLocaleInfoA(LOCALE_USER_DEFAULT,LOCALE_SISO639LANGNAME,buf,sizeof(buf))==0)
return "C";
std::string lc_name = buf;
if(GetLocaleInfoA(LOCALE_USER_DEFAULT,LOCALE_SISO3166CTRYNAME,buf,sizeof(buf))!=0) {
lc_name += "_";
lc_name += buf;
}
if(!use_utf8) {
if(GetLocaleInfoA(LOCALE_USER_DEFAULT,LOCALE_IDEFAULTANSICODEPAGE,buf,sizeof(buf))!=0) {
if(atoi(buf)==0)
lc_name+=".UTF-8";
else {
lc_name +=".windows-";
lc_name +=buf;
}
}
else {
lc_name += "UTF-8";
}
}
else {
lc_name += ".UTF-8";
}
return lc_name;

#endif
}
} 
} 
} 


