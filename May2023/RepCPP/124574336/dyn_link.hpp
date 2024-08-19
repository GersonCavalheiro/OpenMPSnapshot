


#ifndef BOOST_IOSTREAMS_DETAIL_CONFIG_DYN_LINK_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CONFIG_DYN_LINK_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>


#ifdef BOOST_HAS_DECLSPEC 
# if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_IOSTREAMS_DYN_LINK)
#  ifdef BOOST_IOSTREAMS_SOURCE
#   define BOOST_IOSTREAMS_DECL __declspec(dllexport)
#  else
#   define BOOST_IOSTREAMS_DECL __declspec(dllimport)
#  endif  
# endif
#else 
# if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_IOSTREAMS_DYN_LINK)
#  ifdef BOOST_IOSTREAMS_SOURCE
#   define BOOST_IOSTREAMS_DECL BOOST_SYMBOL_EXPORT
#  else
#   define BOOST_IOSTREAMS_DECL BOOST_SYMBOL_IMPORT
#  endif
# endif
#endif 

#ifndef BOOST_IOSTREAMS_DECL
# define BOOST_IOSTREAMS_DECL
#endif

#endif 
