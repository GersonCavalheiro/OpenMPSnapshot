

#ifndef BOOST_IOSTREAMS_OPTIMAL_BUFFER_SIZE_HPP_INCLUDED
#define BOOST_IOSTREAMS_OPTIMAL_BUFFER_SIZE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/constants.hpp>  
#include <boost/iostreams/detail/dispatch.hpp>
#include <boost/iostreams/detail/wrap_unwrap.hpp>
#include <boost/iostreams/operations_fwd.hpp>
#include <boost/mpl/if.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>

namespace boost { namespace iostreams {

namespace detail {

template<typename T>
struct optimal_buffer_size_impl;

} 

template<typename T>
std::streamsize optimal_buffer_size(const T& t)
{
typedef detail::optimal_buffer_size_impl<T> impl;
return impl::optimal_buffer_size(detail::unwrap(t));
}

namespace detail {


template<typename T>
struct optimal_buffer_size_impl
: mpl::if_<
is_custom<T>,
operations<T>,
optimal_buffer_size_impl<
BOOST_DEDUCED_TYPENAME
dispatch<
T, optimally_buffered_tag, device_tag, filter_tag
>::type
>
>::type
{ };

template<>
struct optimal_buffer_size_impl<optimally_buffered_tag> {
template<typename T>
static std::streamsize optimal_buffer_size(const T& t)
{ return t.optimal_buffer_size(); }
};

template<>
struct optimal_buffer_size_impl<device_tag> {
template<typename T>
static std::streamsize optimal_buffer_size(const T&)
{ return default_device_buffer_size; }
};

template<>
struct optimal_buffer_size_impl<filter_tag> {
template<typename T>
static std::streamsize optimal_buffer_size(const T&)
{ return default_filter_buffer_size; }
};

} 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
