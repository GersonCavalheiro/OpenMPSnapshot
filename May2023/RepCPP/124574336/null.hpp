


#ifndef BOOST_IOSTREAMS_NULL_HPP_INCLUDED
#define BOOST_IOSTREAMS_NULL_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/detail/ios.hpp> 
#include <boost/iostreams/positioning.hpp>

namespace boost { namespace iostreams {

template<typename Ch, typename Mode>
class basic_null_device {
public:
typedef Ch char_type;
struct category
: public Mode,
public device_tag,
public closable_tag
{ };
std::streamsize read(Ch*, std::streamsize) { return -1; }
std::streamsize write(const Ch*, std::streamsize n) { return n; }
std::streampos seek( stream_offset, BOOST_IOS::seekdir,
BOOST_IOS::openmode = 
BOOST_IOS::in | BOOST_IOS::out ) 
{ return -1; }
void close() { }
void close(BOOST_IOS::openmode) { }
};

template<typename Ch>
struct basic_null_source : private basic_null_device<Ch, input> {
typedef Ch          char_type;
typedef source_tag  category;
using basic_null_device<Ch, input>::read;
using basic_null_device<Ch, input>::close;
};

typedef basic_null_source<char>     null_source;
typedef basic_null_source<wchar_t>  wnull_source;

template<typename Ch>
struct basic_null_sink : private basic_null_device<Ch, output> {
typedef Ch        char_type;
typedef sink_tag  category;
using basic_null_device<Ch, output>::write;
using basic_null_device<Ch, output>::close;
};

typedef basic_null_sink<char>     null_sink;
typedef basic_null_sink<wchar_t>  wnull_sink;

} } 

#endif 
