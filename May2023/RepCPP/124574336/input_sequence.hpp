

#ifndef BOOST_IOSTREAMS_INPUT_SEQUENCE_HPP_INCLUDED
#define BOOST_IOSTREAMS_INPUT_SEQUENCE_HPP_INCLUDED

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
struct input_sequence_impl;

} 

template<typename T>
inline std::pair<
BOOST_DEDUCED_TYPENAME char_type_of<T>::type*,
BOOST_DEDUCED_TYPENAME char_type_of<T>::type*
>
input_sequence(T& t)
{ return detail::input_sequence_impl<T>::input_sequence(t); }

namespace detail {


template<typename T>
struct input_sequence_impl
: mpl::if_<
detail::is_custom<T>,
operations<T>,
input_sequence_impl<direct_tag>
>::type
{ };

template<>
struct input_sequence_impl<direct_tag> {
template<typename U>
static std::pair<
BOOST_DEDUCED_TYPENAME char_type_of<U>::type*,
BOOST_DEDUCED_TYPENAME char_type_of<U>::type*
>
input_sequence(U& u) { return u.input_sequence(); }
};

} 

} } 

#include <boost/iostreams/detail/config/enable_warnings.hpp>

#endif 
