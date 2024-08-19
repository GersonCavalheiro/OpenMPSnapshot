

#ifndef BOOST_IOSTREAMS_DETAIL_MODE_ADAPTER_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_MODE_ADAPTER_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              


#include <boost/config.hpp>                
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/traits.hpp>
#include <boost/iostreams/operations.hpp> 
#include <boost/mpl/if.hpp> 

namespace boost { namespace iostreams { namespace detail {

template<typename Mode, typename T>
class mode_adapter {
private:
struct empty_base { };
public:
typedef typename wrapped_type<T>::type  component_type;
typedef typename char_type_of<T>::type  char_type;
struct category 
: Mode, 
device_tag,
mpl::if_<is_filter<T>, filter_tag, device_tag>,
mpl::if_<is_filter<T>, multichar_tag, empty_base>,
closable_tag,
localizable_tag
{ };
explicit mode_adapter(const component_type& t) : t_(t) { }


std::streamsize read(char_type* s, std::streamsize n);
std::streamsize write(const char_type* s, std::streamsize n);
std::streampos seek( stream_offset off, BOOST_IOS::seekdir way,
BOOST_IOS::openmode which = 
BOOST_IOS::in | BOOST_IOS::out );
void close();
void close(BOOST_IOS::openmode which);


template<typename Source>
std::streamsize read(Source& src, char_type* s, std::streamsize n)
{ return iostreams::read(t_, src, s, n); }

template<typename Sink>
std::streamsize write(Sink& snk, const char_type* s, std::streamsize n)
{ return iostreams::write(t_, snk, s, n); }

template<typename Device>
std::streampos seek(Device& dev, stream_offset off, BOOST_IOS::seekdir way)
{ return iostreams::seek(t_, dev, off, way); }

template<typename Device>
std::streampos seek( Device& dev, stream_offset off, 
BOOST_IOS::seekdir way, BOOST_IOS::openmode which  )
{ return iostreams::seek(t_, dev, off, way, which); }

template<typename Device>
void close(Device& dev)
{ detail::close_all(t_, dev); }

template<typename Device>
void close(Device& dev, BOOST_IOS::openmode which)
{ iostreams::close(t_, dev, which); }

template<typename Locale>
void imbue(const Locale& loc)
{ iostreams::imbue(t_, loc); }
private:
component_type t_;
};


template<typename Mode, typename T>
std::streamsize mode_adapter<Mode, T>::read
(char_type* s, std::streamsize n)
{ return boost::iostreams::read(t_, s, n); }

template<typename Mode, typename T>
std::streamsize mode_adapter<Mode, T>::write
(const char_type* s, std::streamsize n)
{ return boost::iostreams::write(t_, s, n); }

template<typename Mode, typename T>
std::streampos mode_adapter<Mode, T>::seek
(stream_offset off, BOOST_IOS::seekdir way, BOOST_IOS::openmode which)
{ return boost::iostreams::seek(t_, off, way, which); }

template<typename Mode, typename T>
void mode_adapter<Mode, T>::close()
{ detail::close_all(t_); }

template<typename Mode, typename T>
void mode_adapter<Mode, T>::close(BOOST_IOS::openmode which)
{ iostreams::close(t_, which); }

} } } 

#endif 
