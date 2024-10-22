


#ifndef BOOST_IOSTREAMS_IO_TRAITS_FWD_HPP_INCLUDED
#define BOOST_IOSTREAMS_IO_TRAITS_FWD_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <iosfwd> 

namespace boost { namespace iostreams {      

template<typename T>
struct is_istream;

template<typename T>
struct is_ostream;

template<typename T>
struct is_iostream;

template<typename T>
struct is_streambuf;

template<typename T>
struct is_istringstream;

template<typename T>
struct is_ostringstream;

template<typename T>
struct is_stringstream;

template<typename T>
struct is_stringbuf;

template<typename T>
struct is_ifstream;

template<typename T>
struct is_ofstream;

template<typename T>
struct is_fstream;

template<typename T>
struct is_filebuf;

template<typename T>
struct is_std_io;

template<typename T>
struct is_std_file_device;

template<typename T>
struct is_std_string_device;

template<typename T>
struct char_type_of;

template<typename T>
struct category_of;

template<typename T>
struct int_type_of;

template<typename T>
struct mode_of;

template<typename T>
struct is_device;

template<typename T>
struct is_filter;

template<typename T>
struct is_direct;

namespace detail {

template<typename T>
struct is_boost_stream;

template<typename T>
struct is_boost_stream_buffer;

template<typename T>
struct is_filtering_stream;

template<typename T>
struct is_filtering_streambuf;

template<typename T>
struct is_linked;

template<typename T>
struct is_boost;

} 

} } 

#endif 
