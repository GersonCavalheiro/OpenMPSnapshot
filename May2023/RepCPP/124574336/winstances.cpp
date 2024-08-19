



#define BOOST_REGEX_SOURCE

#include <boost/regex/config.hpp>

#if !defined(BOOST_NO_WREGEX) && !defined(BOOST_REGEX_NO_EXTERNAL_TEMPLATES)
#define BOOST_REGEX_WIDE_INSTANTIATE

#ifdef BOOST_BORLANDC
#pragma hrdstop
#endif

#include <boost/regex.hpp>

#endif



