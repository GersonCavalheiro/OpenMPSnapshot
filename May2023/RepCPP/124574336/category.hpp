
#ifndef BOOST_RANGE_CATEGORY_HPP
#define BOOST_RANGE_CATEGORY_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/iterator.hpp>
#include <boost/iterator/iterator_traits.hpp>

namespace boost
{
template< class T >
struct range_category : iterator_category< typename range_iterator<T>::type >
{ };
}

#endif
