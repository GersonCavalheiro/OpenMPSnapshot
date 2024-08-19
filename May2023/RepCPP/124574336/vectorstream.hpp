


#ifndef BOOST_INTERPROCESS_VECTORSTREAM_HPP
#define BOOST_INTERPROCESS_VECTORSTREAM_HPP

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
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/assert.hpp>

namespace boost {  namespace interprocess {

template <class CharVector, class CharTraits>
class basic_vectorbuf
: public std::basic_streambuf<typename CharVector::value_type, CharTraits>
{
public:
typedef CharVector                        vector_type;
typedef typename CharVector::value_type   char_type;
typedef typename CharTraits::int_type     int_type;
typedef typename CharTraits::pos_type     pos_type;
typedef typename CharTraits::off_type     off_type;
typedef CharTraits                        traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef std::basic_streambuf<char_type, traits_type> base_t;

basic_vectorbuf(const basic_vectorbuf&);
basic_vectorbuf & operator =(const basic_vectorbuf&);
#endif   

public:
explicit basic_vectorbuf(std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
:  base_t(), m_mode(mode)
{  this->initialize_pointers();   }

template<class VectorParameter>
explicit basic_vectorbuf(const VectorParameter &param,
std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
:  base_t(), m_mode(mode), m_vect(param)
{  this->initialize_pointers();   }

public:

void swap_vector(vector_type &vect)
{
if (this->m_mode & std::ios_base::out){
if (mp_high_water < base_t::pptr()){
mp_high_water = base_t::pptr();
}
m_vect.resize(mp_high_water - (m_vect.size() ? &m_vect[0] : 0));
}
m_vect.swap(vect);
this->initialize_pointers();
}

const vector_type &vector() const
{
if (this->m_mode & std::ios_base::out){
if (mp_high_water < base_t::pptr()){
mp_high_water = base_t::pptr();
}
typedef typename vector_type::size_type size_type;
char_type *old_ptr = base_t::pbase();
size_type high_pos = size_type(mp_high_water-old_ptr);
if(m_vect.size() > high_pos){
m_vect.resize(high_pos);
int old_pos = base_t::pptr() - base_t::pbase();
const_cast<basic_vectorbuf*>(this)->base_t::setp(old_ptr, old_ptr + high_pos);
const_cast<basic_vectorbuf*>(this)->base_t::pbump(old_pos);
}
}
return m_vect;
}

void reserve(typename vector_type::size_type size)
{
if (this->m_mode & std::ios_base::out && size > m_vect.size()){
typename vector_type::difference_type write_pos = base_t::pptr() - base_t::pbase();
typename vector_type::difference_type read_pos  = base_t::gptr() - base_t::eback();
m_vect.reserve(size);
this->initialize_pointers();
base_t::pbump((int)write_pos);
if(this->m_mode & std::ios_base::in){
base_t::gbump((int)read_pos);
}
}
}

void clear()
{  m_vect.clear();   this->initialize_pointers();   }

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
void initialize_pointers()
{
if(!(m_mode & std::ios_base::out)){
if(m_vect.empty()){
this->setg(0, 0, 0);
}
else{
this->setg(&m_vect[0], &m_vect[0], &m_vect[0] + m_vect.size());
}
}

if(m_mode & std::ios_base::out){
int real_size = (int)m_vect.size();
m_vect.resize(m_vect.capacity());
BOOST_ASSERT(m_vect.size() == m_vect.capacity());
mp_high_water = m_vect.size() ? (&m_vect[0] + real_size) : 0;
if(m_vect.empty()){
this->setp(0, 0);
if(m_mode & std::ios_base::in)
this->setg(0, 0, 0);
}
else{
char_type *p = &m_vect[0];
this->setp(p, p + m_vect.size());
if(m_mode & std::ios_base::in)
this->setg(p, p, p + real_size);
}
if (m_mode & (std::ios_base::app | std::ios_base::ate)){
base_t::pbump((int)real_size);
}
}
}

protected:
virtual int_type underflow()
{
if (base_t::gptr() == 0)
return CharTraits::eof();
if(m_mode & std::ios_base::out){
if (mp_high_water < base_t::pptr())
mp_high_water = base_t::pptr();
if (base_t::egptr() < mp_high_water)
base_t::setg(base_t::eback(), base_t::gptr(), mp_high_water);
}
if (base_t::gptr() < base_t::egptr())
return CharTraits::to_int_type(*base_t::gptr());
return CharTraits::eof();
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
typedef typename vector_type::difference_type dif_t;
dif_t new_outpos = base_t::pptr() - base_t::pbase() + 1;
dif_t hipos = mp_high_water - base_t::pbase();
if (hipos < new_outpos)
hipos = new_outpos;
m_vect.push_back(CharTraits::to_char_type(c));
m_vect.resize(m_vect.capacity());
BOOST_ASSERT(m_vect.size() == m_vect.capacity());
char_type* p = const_cast<char_type*>(&m_vect[0]);
base_t::setp(p, p + (dif_t)m_vect.size());
mp_high_water = p + hipos;
if (m_mode & std::ios_base::in)
base_t::setg(p, p + (base_t::gptr() - base_t::eback()), mp_high_water);
base_t::pbump((int)new_outpos);
return c;
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
bool in(0 != (mode & std::ios_base::in)), out(0 != (mode & std::ios_base::out));
if(!in & !out)
return pos_type(off_type(-1));
else if((in && out) && (dir == std::ios_base::cur))
return pos_type(off_type(-1));
else if((in  && (!(m_mode & std::ios_base::in) || (off != 0 && this->gptr() == 0) )) ||
(out && (!(m_mode & std::ios_base::out) || (off != 0 && this->pptr() == 0))))
return pos_type(off_type(-1));

off_type newoff;
off_type limit;
if(m_mode & std::ios_base::out){
if(mp_high_water < base_t::pptr())
mp_high_water = base_t::pptr();
if(m_mode & std::ios_base::in){
if (base_t::egptr() < mp_high_water)
base_t::setg(base_t::eback(), base_t::gptr(), mp_high_water);
}
limit = static_cast<off_type>(mp_high_water - base_t::pbase());
}
else{
limit = static_cast<off_type>(m_vect.size());
}

switch(dir) {
case std::ios_base::beg:
newoff = 0;
break;
case std::ios_base::end:
newoff = limit;
break;
case std::ios_base::cur:
newoff = in ? static_cast<std::streamoff>(this->gptr() - this->eback())
: static_cast<std::streamoff>(this->pptr() - this->pbase());
break;
default:
return pos_type(off_type(-1));
}

newoff += off;

if (newoff < 0 || newoff > limit)
return pos_type(-1);
if (m_mode & std::ios_base::app && mode & std::ios_base::out && newoff != limit)
return pos_type(-1);
if (in)
base_t::setg(base_t::eback(), base_t::eback() + newoff, base_t::egptr());
if (out){
base_t::setp(base_t::pbase(), base_t::epptr());
base_t::pbump(newoff);
}
return pos_type(newoff);
}

virtual pos_type seekpos(pos_type pos, std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
{  return seekoff(pos - pos_type(off_type(0)), std::ios_base::beg, mode);  }

private:
std::ios_base::openmode m_mode;
mutable vector_type     m_vect;
mutable char_type*      mp_high_water;
#endif   
};

template <class CharVector, class CharTraits>
class basic_ivectorstream
: public std::basic_istream<typename CharVector::value_type, CharTraits>
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
, private basic_vectorbuf<CharVector, CharTraits>
#endif   
{
public:
typedef CharVector                                                   vector_type;
typedef typename std::basic_ios
<typename CharVector::value_type, CharTraits>::char_type          char_type;
typedef typename std::basic_ios<char_type, CharTraits>::int_type     int_type;
typedef typename std::basic_ios<char_type, CharTraits>::pos_type     pos_type;
typedef typename std::basic_ios<char_type, CharTraits>::off_type     off_type;
typedef typename std::basic_ios<char_type, CharTraits>::traits_type  traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef basic_vectorbuf<CharVector, CharTraits>    vectorbuf_t;
typedef std::basic_ios<char_type, CharTraits>      basic_ios_t;
typedef std::basic_istream<char_type, CharTraits>  base_t;

vectorbuf_t &       get_buf()      {  return *this;  }
const vectorbuf_t & get_buf() const{  return *this;  }
#endif   

public:

basic_ivectorstream(std::ios_base::openmode mode = std::ios_base::in)
: base_t(0) 
, vectorbuf_t(mode | std::ios_base::in)
{  this->base_t::rdbuf(&get_buf()); }

template<class VectorParameter>
basic_ivectorstream(const VectorParameter &param,
std::ios_base::openmode mode = std::ios_base::in)
: vectorbuf_t(param, mode | std::ios_base::in)
, base_t(&get_buf())
{}

public:
basic_vectorbuf<CharVector, CharTraits>* rdbuf() const
{ return const_cast<basic_vectorbuf<CharVector, CharTraits>*>(&get_buf()); }

void swap_vector(vector_type &vect)
{  get_buf().swap_vector(vect);   }

const vector_type &vector() const
{  return get_buf().vector();   }

void reserve(typename vector_type::size_type size)
{  get_buf().reserve(size);   }

void clear()
{  get_buf().clear();   }
};

template <class CharVector, class CharTraits>
class basic_ovectorstream
: public std::basic_ostream<typename CharVector::value_type, CharTraits>
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
, private basic_vectorbuf<CharVector, CharTraits>
#endif   
{
public:
typedef CharVector                                                   vector_type;
typedef typename std::basic_ios
<typename CharVector::value_type, CharTraits>::char_type          char_type;
typedef typename std::basic_ios<char_type, CharTraits>::int_type     int_type;
typedef typename std::basic_ios<char_type, CharTraits>::pos_type     pos_type;
typedef typename std::basic_ios<char_type, CharTraits>::off_type     off_type;
typedef typename std::basic_ios<char_type, CharTraits>::traits_type  traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef basic_vectorbuf<CharVector, CharTraits>    vectorbuf_t;
typedef std::basic_ios<char_type, CharTraits>      basic_ios_t;
typedef std::basic_ostream<char_type, CharTraits>  base_t;

vectorbuf_t &       get_buf()      {  return *this;  }
const vectorbuf_t & get_buf()const {  return *this;  }
#endif   

public:
basic_ovectorstream(std::ios_base::openmode mode = std::ios_base::out)
: base_t(0) 
, vectorbuf_t(mode | std::ios_base::out)
{  this->base_t::rdbuf(&get_buf()); }

template<class VectorParameter>
basic_ovectorstream(const VectorParameter &param,
std::ios_base::openmode mode = std::ios_base::out)
: base_t(0) 
, vectorbuf_t(param, mode | std::ios_base::out)
{  this->base_t::rdbuf(&get_buf()); }

public:
basic_vectorbuf<CharVector, CharTraits>* rdbuf() const
{ return const_cast<basic_vectorbuf<CharVector, CharTraits>*>(&get_buf()); }

void swap_vector(vector_type &vect)
{  get_buf().swap_vector(vect);   }

const vector_type &vector() const
{  return get_buf().vector();   }

void reserve(typename vector_type::size_type size)
{  get_buf().reserve(size);   }
};

template <class CharVector, class CharTraits>
class basic_vectorstream
: public std::basic_iostream<typename CharVector::value_type, CharTraits>
#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
, private basic_vectorbuf<CharVector, CharTraits>
#endif   
{
public:
typedef CharVector                                                   vector_type;
typedef typename std::basic_ios
<typename CharVector::value_type, CharTraits>::char_type          char_type;
typedef typename std::basic_ios<char_type, CharTraits>::int_type     int_type;
typedef typename std::basic_ios<char_type, CharTraits>::pos_type     pos_type;
typedef typename std::basic_ios<char_type, CharTraits>::off_type     off_type;
typedef typename std::basic_ios<char_type, CharTraits>::traits_type  traits_type;

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
private:
typedef basic_vectorbuf<CharVector, CharTraits>    vectorbuf_t;
typedef std::basic_ios<char_type, CharTraits>      basic_ios_t;
typedef std::basic_iostream<char_type, CharTraits> base_t;

vectorbuf_t &       get_buf()      {  return *this;  }
const vectorbuf_t & get_buf() const{  return *this;  }
#endif   

public:
basic_vectorstream(std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
: base_t(0) 
, vectorbuf_t(mode)
{  this->base_t::rdbuf(&get_buf()); }

template<class VectorParameter>
basic_vectorstream(const VectorParameter &param, std::ios_base::openmode mode
= std::ios_base::in | std::ios_base::out)
: base_t(0) 
, vectorbuf_t(param, mode)
{  this->base_t::rdbuf(&get_buf()); }

public:
basic_vectorbuf<CharVector, CharTraits>* rdbuf() const
{ return const_cast<basic_vectorbuf<CharVector, CharTraits>*>(&get_buf()); }

void swap_vector(vector_type &vect)
{  get_buf().swap_vector(vect);   }

const vector_type &vector() const
{  return get_buf().vector();   }

void reserve(typename vector_type::size_type size)
{  get_buf().reserve(size);   }

void clear()
{  get_buf().clear();   }
};


}} 

#include <boost/interprocess/detail/config_end.hpp>

#endif 
