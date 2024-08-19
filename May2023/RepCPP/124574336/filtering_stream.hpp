

#ifndef BOOST_IOSTREAMS_FILTER_STREAM_HPP_INCLUDED
#define BOOST_IOSTREAMS_FILTER_STREAM_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <memory>                                     
#include <boost/iostreams/detail/access_control.hpp>
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/iostream.hpp>        
#include <boost/iostreams/detail/push.hpp>
#include <boost/iostreams/detail/select.hpp>
#include <boost/iostreams/detail/streambuf.hpp>       
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>  

namespace boost { namespace iostreams {


namespace detail {

template<typename Mode, typename Ch, typename Tr>
struct filtering_stream_traits {
typedef typename 
iostreams::select<  
mpl::and_< 
is_convertible<Mode, input>, 
is_convertible<Mode, output> 
>,          
BOOST_IOSTREAMS_BASIC_IOSTREAM(Ch, Tr),
is_convertible<Mode, input>, 
BOOST_IOSTREAMS_BASIC_ISTREAM(Ch, Tr),
else_,        
BOOST_IOSTREAMS_BASIC_OSTREAM(Ch, Tr)
>::type stream_type;
typedef typename
iostreams::select< 
mpl::and_<
is_convertible<Mode, input>,
is_convertible<Mode, output>
>,
iostream_tag,
is_convertible<Mode, input>,
istream_tag,
else_,
ostream_tag
>::type stream_tag;
};

#if defined(BOOST_MSVC) && (BOOST_MSVC == 1700)
# pragma warning(push)
# pragma warning(disable: 4250)
#endif

template<typename Chain, typename Access>
class filtering_stream_base 
: public access_control<
boost::iostreams::detail::chain_client<Chain>,
Access
>,
public filtering_stream_traits<
typename Chain::mode, 
typename Chain::char_type, 
typename Chain::traits_type
>::stream_type
{
public:
typedef Chain                                         chain_type;
typedef access_control<
boost::iostreams::detail::chain_client<Chain>,
Access
>                                            client_type;
protected:
typedef typename 
filtering_stream_traits<
typename Chain::mode, 
typename Chain::char_type, 
typename Chain::traits_type
>::stream_type                                stream_type;
filtering_stream_base() : stream_type(0) { this->set_chain(&chain_); }
private:
void notify() { this->rdbuf(chain_.empty() ? 0 : &chain_.front()); }
Chain chain_;
};

#if defined(BOOST_MSVC) && (BOOST_MSVC == 1700)
# pragma warning(pop)
#endif

} 

#define BOOST_IOSTREAMS_DEFINE_FILTER_STREAM(name_, chain_type_, default_char_) \
template< typename Mode, \
typename Ch = default_char_, \
typename Tr = BOOST_IOSTREAMS_CHAR_TRAITS(Ch), \
typename Alloc = std::allocator<Ch>, \
typename Access = public_ > \
class name_ \
: public boost::iostreams::detail::filtering_stream_base< \
chain_type_<Mode, Ch, Tr, Alloc>, Access \
> \
{ \
public: \
typedef Ch                                char_type; \
struct category \
: Mode, \
closable_tag, \
detail::filtering_stream_traits<Mode, Ch, Tr>::stream_tag \
{ }; \
BOOST_IOSTREAMS_STREAMBUF_TYPEDEFS(Tr) \
typedef Mode                              mode; \
typedef chain_type_<Mode, Ch, Tr, Alloc>  chain_type; \
name_() { } \
BOOST_IOSTREAMS_DEFINE_PUSH_CONSTRUCTOR(name_, mode, Ch, push_impl) \
~name_() { \
if (this->is_complete()) \
this->rdbuf()->BOOST_IOSTREAMS_PUBSYNC(); \
} \
private: \
typedef access_control< \
boost::iostreams::detail::chain_client< \
chain_type_<Mode, Ch, Tr, Alloc> \
>, \
Access \
> client_type; \
template<typename T> \
void push_impl(const T& t BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ client_type::push(t BOOST_IOSTREAMS_PUSH_ARGS()); } \
}; \


#if defined(BOOST_MSVC) && (BOOST_MSVC == 1700)
# pragma warning(push)
# pragma warning(disable: 4250)
#endif

BOOST_IOSTREAMS_DEFINE_FILTER_STREAM(filtering_stream, boost::iostreams::chain, char)
BOOST_IOSTREAMS_DEFINE_FILTER_STREAM(wfiltering_stream, boost::iostreams::chain, wchar_t)

#if defined(BOOST_MSVC) && (BOOST_MSVC == 1700)
# pragma warning(pop)
#endif

typedef filtering_stream<input>    filtering_istream;
typedef filtering_stream<output>   filtering_ostream;
typedef wfiltering_stream<input>   filtering_wistream;
typedef wfiltering_stream<output>  filtering_wostream;


} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp> 

#endif 
