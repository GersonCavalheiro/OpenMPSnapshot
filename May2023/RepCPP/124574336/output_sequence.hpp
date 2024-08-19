

#ifndef BOOST_IOSTREAMS_OUTPUT_SEQUENCE_HPP_INCLUDED
#define BOOST_IOSTREAMS_OUTPUT_SEQUENCE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <utility>           
#include <boost/config.hpp>  
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/wrap_unwrap.hpp>
#include <boost/iostreams/operations_fwd.hpp>  
#include <boost/iostreams/traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/iostreams/detail/config/disable_warnings.hpp>

namespace boost { namespace iostreams {

namespace detail {

template<typename T>
struct output_sequence_impl;

} 

template<typename T>
inline std::pair<
BOOST_DEDUCED_TYPENAME char_type_of<T>::type*,
BOOST_DEDUCED_TYPENAME char_type_of<T>::type*
>
output_sequence(T& t)
{ return detail::output_sequence_impl<T>::output_sequence(t); }

namespace detail {


template<typename T>
struct output_sequence_impl
: mpl::if_<
detail::is_custom<T>,
operations<T>,
output_sequence_impl<direct_tag>
>::type
{ };

template<>
struct output_sequence_impl<direct_tag> {
template<typename U>
static std::pair<
BOOST_DEDUCED_TYPENAME char_type_of<U>::type*,
BOOST_DEDUCED_TYPENAME char_type_of<U>::type*
>
output_sequence(U& u) { return u.output_sequence(); }
};

} 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
