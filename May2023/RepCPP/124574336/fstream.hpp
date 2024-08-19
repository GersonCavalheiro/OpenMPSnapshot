

#ifndef BOOST_IOSTREAMS_DETAIL_FSTREAM_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_FSTREAM_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/iostreams/detail/config/wide_streams.hpp>
#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# include <fstream>
#else
# include <fstream.h>
#endif 

#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# define BOOST_IOSTREAMS_BASIC_IFSTREAM(Ch, Tr) std::basic_ifstream<Ch, Tr>
# define BOOST_IOSTREAMS_BASIC_OFSTREAM(Ch, Tr) std::basic_ofstream<Ch, Tr>
# define BOOST_IOSTREAMS_BASIC_FSTREAM(Ch, Tr) std::basic_fstream<Ch, Tr>
# define BOOST_IOSTREAMS_BASIC_FILEBUF(Ch) std::basic_filebuf<Ch>
#else 
# define BOOST_IOSTREAMS_BASIC_IFSTREAM(Ch, Tr) std::ifstream
# define BOOST_IOSTREAMS_BASIC_OFSTREAM(Ch, Tr) std::ofstream
# define BOOST_IOSTREAMS_BASIC_FILEBUF(Ch) std::filebuf
#endif

#endif 
