

#ifndef BOOST_IOSTREAMS_FILTERING_STREAMBUF_HPP_INCLUDED
#define BOOST_IOSTREAMS_FILTERING_STREAMBUF_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <exception>
#include <memory>                               
#include <boost/iostreams/chain.hpp>
#include <boost/iostreams/detail/access_control.hpp>
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/push.hpp>
#include <boost/iostreams/detail/streambuf.hpp> 
#include <boost/iostreams/detail/streambuf/chainbuf.hpp>
#include <boost/mpl/if.hpp>                    

namespace boost { namespace iostreams {

#define BOOST_IOSTREAMS_DEFINE_FILTER_STREAMBUF(name_, chain_type_, default_char_) \
template< typename Mode, \
typename Ch = default_char_, \
typename Tr = BOOST_IOSTREAMS_CHAR_TRAITS(Ch), \
typename Alloc = std::allocator<Ch>, \
typename Access = public_ > \
class name_ : public boost::iostreams::detail::chainbuf< \
chain_type_<Mode, Ch, Tr, Alloc>, Mode, Access \
> \
{ \
public: \
typedef Ch                                             char_type; \
struct category \
: Mode, closable_tag, streambuf_tag \
{ }; \
BOOST_IOSTREAMS_STREAMBUF_TYPEDEFS(Tr) \
typedef Mode                                           mode; \
typedef chain_type_<Mode, Ch, Tr, Alloc>               chain_type; \
name_() { } \
BOOST_IOSTREAMS_DEFINE_PUSH_CONSTRUCTOR(name_, mode, Ch, push_impl) \
~name_() { if (this->is_complete()) this->BOOST_IOSTREAMS_PUBSYNC(); } \
}; \

BOOST_IOSTREAMS_DEFINE_FILTER_STREAMBUF(filtering_streambuf, boost::iostreams::chain, char)
BOOST_IOSTREAMS_DEFINE_FILTER_STREAMBUF(filtering_wstreambuf, boost::iostreams::chain, wchar_t)

typedef filtering_streambuf<input>    filtering_istreambuf;
typedef filtering_streambuf<output>   filtering_ostreambuf;
typedef filtering_wstreambuf<input>   filtering_wistreambuf;
typedef filtering_wstreambuf<output>  filtering_wostreambuf;

} } 

#endif 
