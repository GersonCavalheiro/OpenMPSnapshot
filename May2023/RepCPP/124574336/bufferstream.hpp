


#ifndef BOOST_INTERPROCESS_BUFFERSTREAM_HPP
#define BOOST_INTERPROCESS_BUFFERSTREAM_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

#include <iosfwd>
#include <ios>
#include <istream>
#include <ostream>
#include <string>    
#include <cstddef>   
#include <boost/assert.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>

namespace boost {  namespace interprocess {

template <class CharT, class CharTraits>
class basic_bufferbuf
: public std::basic_streambuf<CharT, CharTraits>
{
public:
typedef CharT                                         char_type;
typedef typename CharTraits::int_type                 int_type;
typedef typename CharTraits::pos_type                 pos_type;
typedef typename CharTraits::off_type                 off_type;
typedef CharTraits                                    traits_type;
typedef std::basic_streambuf<char_type, traits_type>  basic_streambuf_t;

public:
explicit basic_bufferbuf(std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
:  basic_streambuf_t(), m_mode(mode), m_buffer(0), m_length(0)
{}

explicit basic_bufferbuf(CharT *buf, std::size_t length,
std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
:  basic_streambuf_t(), m_mode(mode), m_buffer(buf), m_length(length)
{  this->set_pointers();   }

virtual ~basic_bufferbuf(){}

public:
std::pair<CharT *, std::size_t> buffer() const
{ return std::pair<CharT *, std::size_t>(m_buffer, m_length); }

void buffer(CharT *buf, std::size_t length)
{  m_buffer = buf;   m_length = length;   this->set_pointers();   }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
void set_pointers()
{
if(m_mode & std::ios_base::in)
this->setg(m_buffer, m_buffer, m_buffer + m_length);

if(m_mode & std::ios_base::out)
this->setp(m_buffer, m_buffer + m_length);
}

protected:
virtual int_type underflow()
{
return this->gptr() != this->egptr() ?
CharTraits::to_int_type(*this->gptr()) : CharTraits::eof();
}

virtual int_type pbackfail(int_type c = CharTraits::eof())
{
if(this->gptr() != this->eback()) {
if(!CharTraits::eq_int_type(c, CharTraits::eof())) {
if(CharTraits::eq(CharTraits::to_char_type(c), this->gptr()[-1])) {
this->gbump(-1);
return c;
}
else if(m_mode & std::ios_base::out) {
this->gbump(-1);
*this->gptr() = c;
return c;
}
else
return CharTraits::eof();
}
else {
this->gbump(-1);
return CharTraits::not_eof(c);
}
}
else
return CharTraits::eof();
}

virtual int_type overflow(int_type c = CharTraits::eof())
{
if(m_mode & std::ios_base::out) {
if(!CharTraits::eq_int_type(c, CharTraits::eof())) {
if(this->pptr() == this->epptr()) {
return CharTraits::eof();
}
else {
*this->pptr() = CharTraits::to_char_type(c);
this->pbump(1);
return c;
}
}
else  
return CharTraits::not_eof(c);
}
else     
return CharTraits::eof();
}

virtual pos_type seekoff(off_type off, std::ios_base::seekdir dir,
std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
{
bool in  = false;
bool out = false;

const std::ios_base::openmode inout =
std::ios_base::in | std::ios_base::out;

if((mode & inout) == inout) {
if(dir == std::ios_base::beg || dir == std::ios_base::end)
in = out = true;
}
else if(mode & std::ios_base::in)
in = true;
else if(mode & std::ios_base::out)
out = true;

if(!in && !out)
return pos_type(off_type(-1));
else if((in  && (!(m_mode & std::ios_base::in) || (off != 0 && this->gptr() == 0) )) ||
(out && (!(m_mode & std::ios_base::out) || (off != 0 && this->pptr() == 0))))
return pos_type(off_type(-1));

std::streamoff newoff;
switch(dir) {
case std::ios_base::beg:
newoff = 0;
break;
case std::ios_base::end:
newoff = static_cast<std::streamoff>(m_length);
break;
case std::ios_base::cur:
newoff = in ? static_cast<std::streamoff>(this->gptr() - this->eback())
: static_cast<std::streamoff>(this->pptr() - this->pbase());
break;
default:
return pos_type(off_type(-1));
}

off += newoff;

if(in) {
std::ptrdiff_t n = this->egptr() - this->eback();

if(off < 0 || off > n)
return pos_type(off_type(-1));
else
this->setg(this->eback(), this->eback() + off, this->eback() + n);
}

if(out) {
std::ptrdiff_t n = this->epptr() - this->pbase();

if(off < 0 || off > n)
return pos_type(off_type(-1));
else {
this->setp(this->pbase(), this->pbase() + n);
this->pbump(static_cast<int>(off));
}
}

return pos_type(off);
}

virtual pos_type seekpos(pos_type pos, std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
{  return seekoff(pos - pos_type(off_type(0)), std::ios_base::beg, mode);  }

private:
std::ios_base::openmode m_mode;
CharT *                 m_buffer;
std::size_t             m_length;
#endif   
};

template <class CharT, class CharTraits>
class basic_ibufferstream :
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private basic_bufferbuf<CharT, CharTraits>,
#endif   
public std::basic_istream<CharT, CharTraits>
{
public:                         
typedef typename std::basic_ios
<CharT, CharTraits>::char_type          char_type;
typedef typename std::basic_ios<char_type, CharTraits>::int_type     int_type;
typedef typename std::basic_ios<char_type, CharTraits>::pos_type     pos_type;
typedef typename std::basic_ios<char_type, CharTraits>::off_type     off_type;
typedef typename std::basic_ios<char_type, CharTraits>::traits_type  traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef basic_bufferbuf<CharT, CharTraits>         bufferbuf_t;
typedef std::basic_ios<char_type, CharTraits>      basic_ios_t;
typedef std::basic_istream<char_type, CharTraits>  basic_streambuf_t;
bufferbuf_t &       get_buf()      {  return *this;  }
const bufferbuf_t & get_buf() const{  return *this;  }
#endif   

public:
basic_ibufferstream(std::ios_base::openmode mode = std::ios_base::in)
:  
bufferbuf_t(mode | std::ios_base::in)
, basic_streambuf_t(this)
{}

basic_ibufferstream(const CharT *buf, std::size_t length,
std::ios_base::openmode mode = std::ios_base::in)
:  
bufferbuf_t(const_cast<CharT*>(buf), length, mode | std::ios_base::in)
, basic_streambuf_t(this)
{}

~basic_ibufferstream(){}

public:
basic_bufferbuf<CharT, CharTraits>* rdbuf() const
{ return const_cast<basic_bufferbuf<CharT, CharTraits>*>(&get_buf()); }

std::pair<const CharT *, std::size_t> buffer() const
{ return get_buf().buffer(); }

void buffer(const CharT *buf, std::size_t length)
{  get_buf().buffer(const_cast<CharT*>(buf), length);  }
};

template <class CharT, class CharTraits>
class basic_obufferstream :
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private basic_bufferbuf<CharT, CharTraits>,
#endif   
public std::basic_ostream<CharT, CharTraits>
{
public:
typedef typename std::basic_ios
<CharT, CharTraits>::char_type          char_type;
typedef typename std::basic_ios<char_type, CharTraits>::int_type     int_type;
typedef typename std::basic_ios<char_type, CharTraits>::pos_type     pos_type;
typedef typename std::basic_ios<char_type, CharTraits>::off_type     off_type;
typedef typename std::basic_ios<char_type, CharTraits>::traits_type  traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef basic_bufferbuf<CharT, CharTraits>         bufferbuf_t;
typedef std::basic_ios<char_type, CharTraits>      basic_ios_t;
typedef std::basic_ostream<char_type, CharTraits>  basic_ostream_t;
bufferbuf_t &       get_buf()      {  return *this;  }
const bufferbuf_t & get_buf() const{  return *this;  }
#endif   

public:
basic_obufferstream(std::ios_base::openmode mode = std::ios_base::out)
:  
bufferbuf_t(mode | std::ios_base::out)
,  basic_ostream_t(this)
{}

basic_obufferstream(CharT *buf, std::size_t length,
std::ios_base::openmode mode = std::ios_base::out)
:  
bufferbuf_t(buf, length, mode | std::ios_base::out)
,  basic_ostream_t(this)
{}

~basic_obufferstream(){}

public:
basic_bufferbuf<CharT, CharTraits>* rdbuf() const
{ return const_cast<basic_bufferbuf<CharT, CharTraits>*>(&get_buf()); }

std::pair<CharT *, std::size_t> buffer() const
{ return get_buf().buffer(); }

void buffer(CharT *buf, std::size_t length)
{  get_buf().buffer(buf, length);  }
};


template <class CharT, class CharTraits>
class basic_bufferstream :
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private basic_bufferbuf<CharT, CharTraits>,
#endif   
public std::basic_iostream<CharT, CharTraits>
{
public:                         
typedef typename std::basic_ios
<CharT, CharTraits>::char_type          char_type;
typedef typename std::basic_ios<char_type, CharTraits>::int_type     int_type;
typedef typename std::basic_ios<char_type, CharTraits>::pos_type     pos_type;
typedef typename std::basic_ios<char_type, CharTraits>::off_type     off_type;
typedef typename std::basic_ios<char_type, CharTraits>::traits_type  traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef basic_bufferbuf<CharT, CharTraits>         bufferbuf_t;
typedef std::basic_ios<char_type, CharTraits>      basic_ios_t;
typedef std::basic_iostream<char_type, CharTraits> basic_iostream_t;
bufferbuf_t &       get_buf()      {  return *this;  }
const bufferbuf_t & get_buf() const{  return *this;  }
#endif   

public:
basic_bufferstream(std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
:  
bufferbuf_t(mode)
,  basic_iostream_t(this)
{}

basic_bufferstream(CharT *buf, std::size_t length,
std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
:  
bufferbuf_t(buf, length, mode)
,  basic_iostream_t(this)
{}

~basic_bufferstream(){}

public:
basic_bufferbuf<CharT, CharTraits>* rdbuf() const
{ return const_cast<basic_bufferbuf<CharT, CharTraits>*>(&get_buf()); }

std::pair<CharT *, std::size_t> buffer() const
{ return get_buf().buffer(); }

void buffer(CharT *buf, std::size_t length)
{  get_buf().buffer(buf, length);  }
};

typedef basic_bufferbuf<char>        bufferbuf;
typedef basic_bufferstream<char>     bufferstream;
typedef basic_ibufferstream<char>    ibufferstream;
typedef basic_obufferstream<char>    obufferstream;

typedef basic_bufferbuf<wchar_t>     wbufferbuf;
typedef basic_bufferstream<wchar_t>  wbufferstream;
typedef basic_ibufferstream<wchar_t> wibufferstream;
typedef basic_obufferstream<wchar_t> wobufferstream;


}} 

#include <boost/interprocess/detail/config_end.hpp>

#endif 
