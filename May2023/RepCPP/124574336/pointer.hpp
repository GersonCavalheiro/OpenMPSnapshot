
#ifndef BOOST_RANGE_POINTER_TYPE_HPP
#define BOOST_RANGE_POINTER_TYPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/iterator.hpp>
#include <boost/iterator/iterator_traits.hpp>

namespace boost
{
template< class T >
struct range_pointer
: iterator_pointer< BOOST_DEDUCED_TYPENAME range_iterator<T>::type >
{ };
}

#endif
