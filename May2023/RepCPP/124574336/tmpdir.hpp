#ifndef BOOST_ARCHIVE_TMPDIR_HPP
#define BOOST_ARCHIVE_TMPDIR_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <cstdlib> 
#include <cstddef> 

#include <boost/config.hpp>
#ifdef BOOST_NO_STDC_NAMESPACE
namespace std {
using ::getenv;
}
#endif

namespace boost {
namespace archive {

inline const char * tmpdir(){
const char *dirname;
dirname = std::getenv("TMP");
if(NULL == dirname)
dirname = std::getenv("TMPDIR");
if(NULL == dirname)
dirname = std::getenv("TEMP");
if(NULL == dirname){
dirname = ".";
}
return dirname;
}

} 
} 

#endif 
