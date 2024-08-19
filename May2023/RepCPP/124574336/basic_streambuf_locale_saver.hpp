#ifndef BOOST_ARCHIVE_BASIC_STREAMBUF_LOCALE_SAVER_HPP
#define BOOST_ARCHIVE_BASIC_STREAMBUF_LOCALE_SAVER_HPP

#if defined(_MSC_VER)
# pragma once
#endif







#ifndef BOOST_NO_STD_LOCALE

#include <locale>     
#include <ios>
#include <streambuf>  

#include <boost/config.hpp>
#include <boost/noncopyable.hpp>

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost{
namespace archive{

template < typename Ch, class Tr >
class basic_streambuf_locale_saver :
private boost::noncopyable
{
public:
explicit basic_streambuf_locale_saver(std::basic_streambuf<Ch, Tr> &s) :
m_streambuf(s),
m_locale(s.getloc())
{}
~basic_streambuf_locale_saver(){
m_streambuf.pubsync();
m_streambuf.pubimbue(m_locale);
}
private:
std::basic_streambuf<Ch, Tr> &       m_streambuf;
std::locale const  m_locale;
};

template < typename Ch, class Tr >
class basic_istream_locale_saver :
private boost::noncopyable
{
public:
explicit basic_istream_locale_saver(std::basic_istream<Ch, Tr> &s) :
m_istream(s),
m_locale(s.getloc())
{}
~basic_istream_locale_saver(){
m_istream.sync();
m_istream.imbue(m_locale);
}
private:
std::basic_istream<Ch, Tr> & m_istream;
std::locale const  m_locale;
};

template < typename Ch, class Tr >
class basic_ostream_locale_saver :
private boost::noncopyable
{
public:
explicit basic_ostream_locale_saver(std::basic_ostream<Ch, Tr> &s) :
m_ostream(s),
m_locale(s.getloc())
{}
~basic_ostream_locale_saver(){
m_ostream.flush();
m_ostream.imbue(m_locale);
}
private:
std::basic_ostream<Ch, Tr> & m_ostream;
std::locale const  m_locale;
};


} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
#endif 
