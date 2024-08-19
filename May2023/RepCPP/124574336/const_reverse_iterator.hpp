
#ifndef BOOST_RANGE_CONST_REVERSE_ITERATOR_HPP
#define BOOST_RANGE_CONST_REVERSE_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config/header_deprecated.hpp>

BOOST_HEADER_DEPRECATED("<boost/range/reverse_iterator.hpp>")

#include <boost/range/reverse_iterator.hpp>
#include <boost/type_traits/remove_reference.hpp>

namespace boost
{

template< typename C >
struct range_const_reverse_iterator
: range_reverse_iterator<
const BOOST_DEDUCED_TYPENAME remove_reference<C>::type>
{ };

} 

#endif
