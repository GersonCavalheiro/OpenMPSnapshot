

#ifndef BOOST_IOSTREAMS_DETAIL_STREAMBUF_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_STREAMBUF_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/iostreams/detail/config/wide_streams.hpp>
#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# include <streambuf>
#else 
# include <streambuf.h>
#endif 

#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# define BOOST_IOSTREAMS_BASIC_STREAMBUF(ch, tr) std::basic_streambuf< ch, tr >
# define BOOST_IOSTREAMS_PUBSYNC pubsync
# define BOOST_IOSTREAMS_PUBSEEKOFF pubseekoff
# define BOOST_IOSTREAMS_PUBSEEKPOS pubseekpos
#else
# define BOOST_IOSTREAMS_BASIC_STREAMBUF(ch, tr) std::streambuf
# define BOOST_IOSTREAMS_PUBSYNC sync
# define BOOST_IOSTREAMS_PUBSEEKOFF seekoff
# define BOOST_IOSTREAMS_PUBSEEKPOS seekpos
#endif

#endif 
