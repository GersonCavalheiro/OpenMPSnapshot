

#ifndef BOOST_IOSTREAMS_OPERATIONS_FWD_HPP_INCLUDED
#define BOOST_IOSTREAMS_OPERATIONS_FWD_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/not.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

namespace boost { namespace iostreams {

template<typename T>
struct operations;

namespace detail {

struct custom_tag { };

template<typename T>
struct is_custom
: mpl::not_<
is_base_and_derived< custom_tag, operations<T> >
>
{ };

} 

template<typename T>
struct operations : detail::custom_tag { };

} } 

#endif 
