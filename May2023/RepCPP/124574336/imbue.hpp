

#ifndef BOOST_IOSTREAMS_IMBUE_HPP_INCLUDED
#define BOOST_IOSTREAMS_IMBUE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/dispatch.hpp>
#include <boost/iostreams/detail/streambuf.hpp>
#include <boost/iostreams/detail/wrap_unwrap.hpp>
#include <boost/iostreams/operations_fwd.hpp>
#include <boost/mpl/if.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>

namespace boost { namespace iostreams { 

namespace detail {

template<typename T> 
struct imbue_impl;

} 

template<typename T, typename Locale>
void imbue(T& t, const Locale& loc)
{ detail::imbue_impl<T>::imbue(detail::unwrap(t), loc); }

namespace detail {


template<typename T>
struct imbue_impl
: mpl::if_<
is_custom<T>,
operations<T>,
imbue_impl<
BOOST_DEDUCED_TYPENAME
dispatch<
T, streambuf_tag, localizable_tag, any_tag
>::type
>
>::type
{ };

template<>
struct imbue_impl<any_tag> {
template<typename T, typename Locale>
static void imbue(T&, const Locale&) { }
};

template<>
struct imbue_impl<streambuf_tag> {
template<typename T, typename Locale>
static void imbue(T& t, const Locale& loc) { t.pubimbue(loc); }
};

template<>
struct imbue_impl<localizable_tag> {
template<typename T, typename Locale>
static void imbue(T& t, const Locale& loc) { t.imbue(loc); }
};

} 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
