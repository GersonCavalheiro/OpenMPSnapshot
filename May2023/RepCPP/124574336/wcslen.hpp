#ifndef BOOST_ARCHIVE_WCSLEN_HPP
#define BOOST_ARCHIVE_WCSLEN_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstddef> 
#include <boost/config.hpp>
#if defined(BOOST_NO_STDC_NAMESPACE)
namespace std{
using ::size_t;
} 
#endif

#ifndef BOOST_NO_CWCHAR


#if defined(BOOST_DINKUMWARE_STDLIB) && BOOST_DINKUMWARE_STDLIB < 306 \
|| defined(__LIBCOMO__)

namespace std {
inline std::size_t wcslen(const wchar_t * ws)
{
const wchar_t * eows = ws;
while(* eows != 0)
++eows;
return eows - ws;
}
} 

#else

#ifndef BOOST_NO_CWCHAR
#include <cwchar>
#endif
#ifdef BOOST_NO_STDC_NAMESPACE
namespace std{ using ::wcslen; }
#endif

#endif 

#endif 

#endif 
