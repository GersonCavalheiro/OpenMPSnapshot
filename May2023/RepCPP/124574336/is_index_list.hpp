

#ifndef BOOST_MULTI_INDEX_DETAIL_IS_INDEX_LIST_HPP
#define BOOST_MULTI_INDEX_DETAIL_IS_INDEX_LIST_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/empty.hpp>
#include <boost/mpl/is_sequence.hpp>

namespace boost{

namespace multi_index{

namespace detail{

template<typename T>
struct is_index_list
{
BOOST_STATIC_CONSTANT(bool,mpl_sequence=mpl::is_sequence<T>::value);
BOOST_STATIC_CONSTANT(bool,non_empty=!mpl::empty<T>::value);
BOOST_STATIC_CONSTANT(bool,value=mpl_sequence&&non_empty);
};

} 

} 

} 

#endif
