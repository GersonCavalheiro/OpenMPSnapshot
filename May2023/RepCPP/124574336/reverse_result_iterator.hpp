
#ifndef BOOST_RANGE_REVERSE_RESULT_ITERATOR_HPP
#define BOOST_RANGE_REVERSE_RESULT_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config/header_deprecated.hpp>

BOOST_HEADER_DEPRECATED("<boost/range/reverse_iterator.hpp>")

#include <boost/range/reverse_iterator.hpp>

namespace boost
{

template< typename C >
struct range_reverse_result_iterator : range_reverse_iterator<C>
{ };

} 

#endif
