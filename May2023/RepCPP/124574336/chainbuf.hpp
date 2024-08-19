

#ifndef BOOST_IOSTREAMS_DETAIL_CHAINBUF_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CHAINBUF_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif      

#include <boost/config.hpp>                    
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/chain.hpp>
#include <boost/iostreams/detail/access_control.hpp>
#include <boost/iostreams/detail/config/wide_streams.hpp>
#include <boost/iostreams/detail/streambuf.hpp>
#include <boost/iostreams/detail/streambuf/linked_streambuf.hpp>
#include <boost/iostreams/detail/translate_int_type.hpp>
#include <boost/iostreams/traits.hpp>
#include <boost/noncopyable.hpp>

namespace boost { namespace iostreams { namespace detail {


template<typename Chain, typename Mode, typename Access>
class chainbuf
: public BOOST_IOSTREAMS_BASIC_STREAMBUF(
typename Chain::char_type,
typename Chain::traits_type
),
public access_control<typename Chain::client_type, Access>,
private noncopyable
{
private:
typedef access_control<chain_client<Chain>, Access>      client_type;
public:
typedef typename Chain::char_type                        char_type;
BOOST_IOSTREAMS_STREAMBUF_TYPEDEFS(typename Chain::traits_type)
protected:
typedef linked_streambuf<char_type, traits_type>         delegate_type;
chainbuf() { client_type::set_chain(&chain_); }
int_type underflow() 
{ sentry t(this); return translate(delegate().underflow()); }
int_type pbackfail(int_type c)
{ sentry t(this); return translate(delegate().pbackfail(c)); }
std::streamsize xsgetn(char_type* s, std::streamsize n)
{ sentry t(this); return delegate().xsgetn(s, n); }
int_type overflow(int_type c)
{ sentry t(this); return translate(delegate().overflow(c)); }
std::streamsize xsputn(const char_type* s, std::streamsize n)
{ sentry t(this); return delegate().xsputn(s, n); }
int sync() { sentry t(this); return delegate().sync(); }
pos_type seekoff( off_type off, BOOST_IOS::seekdir way,
BOOST_IOS::openmode which =
BOOST_IOS::in | BOOST_IOS::out )
{ sentry t(this); return delegate().seekoff(off, way, which); }
pos_type seekpos( pos_type sp,
BOOST_IOS::openmode which =
BOOST_IOS::in | BOOST_IOS::out )
{ sentry t(this); return delegate().seekpos(sp, which); }
protected:
typedef BOOST_IOSTREAMS_BASIC_STREAMBUF(
typename Chain::char_type,
typename Chain::traits_type
)                                               base_type;
private:

typedef BOOST_IOSTREAMS_CHAR_TRAITS(char_type)           std_traits;
typedef typename Chain::traits_type                      chain_traits;
static typename chain_traits::int_type 
translate(typename std_traits::int_type c)
{ return translate_int_type<std_traits, chain_traits>(c); }

delegate_type& delegate() 
{ return static_cast<delegate_type&>(chain_.front()); }
void get_pointers()
{
this->setg(delegate().eback(), delegate().gptr(), delegate().egptr());
this->setp(delegate().pbase(), delegate().epptr());
this->pbump((int) (delegate().pptr() - delegate().pbase()));
}
void set_pointers()
{
delegate().setg(this->eback(), this->gptr(), this->egptr());
delegate().setp(this->pbase(), this->epptr());
delegate().pbump((int) (this->pptr() - this->pbase()));
}
struct sentry {
sentry(chainbuf<Chain, Mode, Access>* buf) : buf_(buf)
{ buf_->set_pointers(); }
~sentry() { buf_->get_pointers(); }
chainbuf<Chain, Mode, Access>* buf_;
};
friend struct sentry;
Chain chain_;
};

} } } 

#endif 
