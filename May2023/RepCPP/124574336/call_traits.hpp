

#ifndef BOOST_IOSTREAMS_DETAIL_VALUE_TYPE_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_VALUE_TYPE_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/iostreams/traits.hpp>
#include <boost/mpl/if.hpp>

namespace boost { namespace iostreams { namespace detail {

template<typename T>
struct param_type {
typedef typename mpl::if_<is_std_io<T>, T&, const T&>::type type;
};

template<typename T>
struct value_type {
typedef typename mpl::if_<is_std_io<T>, T&, T>::type type;
};

} } } 

#endif 
