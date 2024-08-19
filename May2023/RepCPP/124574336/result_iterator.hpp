
#ifndef BOOST_RANGE_RESULT_ITERATOR_HPP
#define BOOST_RANGE_RESULT_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config/header_deprecated.hpp>

BOOST_HEADER_DEPRECATED("<boost/range/iterator.hpp>")

#include <boost/range/iterator.hpp>

namespace boost
{

template< typename C >
struct range_result_iterator : range_iterator<C>
{ };

} 


#endif
