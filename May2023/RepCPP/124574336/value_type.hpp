
#ifndef BOOST_RANGE_VALUE_TYPE_HPP
#define BOOST_RANGE_VALUE_TYPE_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/iterator.hpp>

#include <boost/iterator/iterator_traits.hpp>

namespace boost
{
template< class T >
struct range_value : iterator_value< typename range_iterator<T>::type >
{ };
}

#endif
