

#ifndef BOOST_IOSTREAMS_STREAM_BUFFER_HPP_INCLUDED
#define BOOST_IOSTREAMS_STREAM_BUFFER_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <memory>            
#include <boost/config.hpp>  
#include <boost/iostreams/detail/char_traits.hpp>
#include <boost/iostreams/detail/config/overload_resolution.hpp>
#include <boost/iostreams/detail/forward.hpp>
#include <boost/iostreams/detail/ios.hpp>  
#include <boost/iostreams/detail/streambuf/direct_streambuf.hpp>
#include <boost/iostreams/detail/streambuf/indirect_streambuf.hpp>
#include <boost/iostreams/traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/is_convertible.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>  

namespace boost { namespace iostreams { namespace detail {

template<typename T, typename Tr, typename Alloc, typename Mode>
struct stream_buffer_traits {
typedef typename
mpl::if_<
is_convertible<
BOOST_DEDUCED_TYPENAME category_of<T>::type,
direct_tag
>,
direct_streambuf<T, Tr>,
indirect_streambuf<T, Tr, Alloc, Mode>
>::type type;
};

} } } 

#ifdef BOOST_IOSTREAMS_BROKEN_OVERLOAD_RESOLUTION
# include <boost/iostreams/detail/broken_overload_resolution/stream_buffer.hpp>
#else

namespace boost { namespace iostreams {

template< typename T,
typename Tr =
BOOST_IOSTREAMS_CHAR_TRAITS(
BOOST_DEDUCED_TYPENAME char_type_of<T>::type
),
typename Alloc =
std::allocator<
BOOST_DEDUCED_TYPENAME char_type_of<T>::type
>,
typename Mode = BOOST_DEDUCED_TYPENAME mode_of<T>::type >
class stream_buffer
: public detail::stream_buffer_traits<T, Tr, Alloc, Mode>::type
{
private:
BOOST_STATIC_ASSERT((
is_convertible<
BOOST_DEDUCED_TYPENAME iostreams::category_of<T>::type, Mode
>::value
));
typedef typename
detail::stream_buffer_traits<
T, Tr, Alloc, Mode
>::type                           base_type;
public:
typedef typename char_type_of<T>::type    char_type;
struct category 
: Mode,
closable_tag,
streambuf_tag
{ };
BOOST_IOSTREAMS_STREAMBUF_TYPEDEFS(Tr)
public:
stream_buffer() { }
~stream_buffer()
{ 
try { 
if (this->is_open() && this->auto_close()) 
this->close(); 
} catch (...) { } 
}
BOOST_IOSTREAMS_FORWARD( stream_buffer, open_impl, T,
BOOST_IOSTREAMS_PUSH_PARAMS,
BOOST_IOSTREAMS_PUSH_ARGS )
T& operator*() { return *this->component(); }
T* operator->() { return this->component(); }
private:
void open_impl(const T& t BOOST_IOSTREAMS_PUSH_PARAMS())
{   
if (this->is_open())
boost::throw_exception(
BOOST_IOSTREAMS_FAILURE("already open")
);
base_type::open(t BOOST_IOSTREAMS_PUSH_ARGS());
}
};

} } 

#endif 

#include <boost/iostreams/detail/config/enable_warnings.hpp>  

#endif 
