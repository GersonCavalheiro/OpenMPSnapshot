

#ifndef BOOST_IOSTREAMS_DETAIL_ERROR_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_ERROR_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/iostreams/detail/ios.hpp>  

namespace boost { namespace iostreams { namespace detail {

inline BOOST_IOSTREAMS_FAILURE cant_read() 
{ return BOOST_IOSTREAMS_FAILURE("no read access"); }

inline BOOST_IOSTREAMS_FAILURE cant_write() 
{ return BOOST_IOSTREAMS_FAILURE("no write access"); }

inline BOOST_IOSTREAMS_FAILURE cant_seek() 
{ return BOOST_IOSTREAMS_FAILURE("no random access"); }

inline BOOST_IOSTREAMS_FAILURE bad_read() 
{ return BOOST_IOSTREAMS_FAILURE("bad read"); }

inline BOOST_IOSTREAMS_FAILURE bad_putback() 
{ return BOOST_IOSTREAMS_FAILURE("putback buffer full"); }

inline BOOST_IOSTREAMS_FAILURE bad_write() 
{ return BOOST_IOSTREAMS_FAILURE("bad write"); }

inline BOOST_IOSTREAMS_FAILURE write_area_exhausted() 
{ return BOOST_IOSTREAMS_FAILURE("write area exhausted"); }

inline BOOST_IOSTREAMS_FAILURE bad_seek() 
{ return BOOST_IOSTREAMS_FAILURE("bad seek"); }

} } } 

#endif 
