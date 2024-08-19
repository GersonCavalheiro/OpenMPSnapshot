
#ifndef BOOST_RANGE_REVERSE_ITERATOR_HPP
#define BOOST_RANGE_REVERSE_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/iterator.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/iterator/reverse_iterator.hpp>


namespace boost
{

template< typename T >
struct range_reverse_iterator
{
typedef reverse_iterator< 
BOOST_DEDUCED_TYPENAME range_iterator<
BOOST_DEDUCED_TYPENAME remove_reference<T>::type>::type > type;
};


} 


#endif
