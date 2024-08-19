

#ifndef BOOST_IOSTREAMS_DETAIL_IOSTREAM_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_IOSTREAM_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/iostreams/detail/config/wide_streams.hpp>
#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# include <istream>
# include <ostream>
#else
# include <iostream.h>
#endif

#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# define BOOST_IOSTREAMS_BASIC_ISTREAM(ch, tr) std::basic_istream< ch, tr >
# define BOOST_IOSTREAMS_BASIC_OSTREAM(ch, tr) std::basic_ostream< ch, tr >
# define BOOST_IOSTREAMS_BASIC_IOSTREAM(ch, tr) std::basic_iostream< ch, tr >
#else
# define BOOST_IOSTREAMS_BASIC_STREAMBUF(ch, tr) std::streambuf
# define BOOST_IOSTREAMS_BASIC_ISTREAM(ch, tr) std::istream
# define BOOST_IOSTREAMS_BASIC_OSTREAM(ch, tr) std::ostream
# define BOOST_IOSTREAMS_BASIC_IOSTREAM(ch, tr) std::iostream
#endif

#endif 
