

#ifndef BOOST_IOSTREAMS_DETAIL_ENABLE_IF_STREAM_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_ENABLE_IF_STREAM_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/config.hpp>                
#include <boost/config/workaround.hpp>
#include <boost/utility/enable_if.hpp>                  
#include <boost/iostreams/traits_fwd.hpp>  

#if !defined(BOOST_NO_SFINAE) && \
!BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x592))
# define BOOST_IOSTREAMS_ENABLE_IF_STREAM(T) \
, typename boost::enable_if< boost::iostreams::is_std_io<T> >::type* = 0 \

# define BOOST_IOSTREAMS_DISABLE_IF_STREAM(T) \
, typename boost::disable_if< boost::iostreams::is_std_io<T> >::type* = 0 \

#else 
# define BOOST_IOSTREAMS_ENABLE_IF_STREAM(T)
# define BOOST_IOSTREAMS_DISABLE_IF_STREAM(T)
#endif

#endif 
