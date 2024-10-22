

#ifndef BOOST_IOSTREAMS_FLUSH_HPP_INCLUDED
#define BOOST_IOSTREAMS_FLUSH_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/dispatch.hpp>
#include <boost/iostreams/detail/streambuf.hpp>
#include <boost/iostreams/detail/wrap_unwrap.hpp>
#include <boost/iostreams/operations_fwd.hpp>
#include <boost/iostreams/traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>

namespace boost { namespace iostreams {

namespace detail {

template<typename T>
struct flush_device_impl;

template<typename T>
struct flush_filter_impl;

} 

template<typename T>
bool flush(T& t)
{ return detail::flush_device_impl<T>::flush(detail::unwrap(t)); }

template<typename T, typename Sink>
bool flush(T& t, Sink& snk)
{ return detail::flush_filter_impl<T>::flush(detail::unwrap(t), snk); }

namespace detail {


template<typename T>
struct flush_device_impl
: mpl::if_<
is_custom<T>,
operations<T>,
flush_device_impl<
BOOST_DEDUCED_TYPENAME
dispatch<
T, ostream_tag, streambuf_tag, flushable_tag, any_tag
>::type
>
>::type
{ };

template<>
struct flush_device_impl<ostream_tag> {
template<typename T>
static bool flush(T& t)
{ return t.rdbuf()->BOOST_IOSTREAMS_PUBSYNC() == 0; }
};

template<>
struct flush_device_impl<streambuf_tag> {
template<typename T>
static bool flush(T& t)
{ return t.BOOST_IOSTREAMS_PUBSYNC() == 0; }
};

template<>
struct flush_device_impl<flushable_tag> {
template<typename T>
static bool flush(T& t) { return t.flush(); }
};

template<>
struct flush_device_impl<any_tag> {
template<typename T>
static bool flush(T&) { return true; }
};


template<typename T>
struct flush_filter_impl
: mpl::if_<
is_custom<T>,
operations<T>,
flush_filter_impl<
BOOST_DEDUCED_TYPENAME
dispatch<
T, flushable_tag, any_tag
>::type
>
>::type
{ };

template<>
struct flush_filter_impl<flushable_tag> {
template<typename T, typename Sink>
static bool flush(T& t, Sink& snk) { return t.flush(snk); }
};

template<>
struct flush_filter_impl<any_tag> {
template<typename T, typename Sink>
static bool flush(T&, Sink&) { return false; }
};

} 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
