
#ifndef BOOST_RANGE_CONST_ITERATOR_HPP
#define BOOST_RANGE_CONST_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>

#include <boost/range/range_fwd.hpp>
#include <boost/range/detail/extract_optional_type.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <cstddef>
#include <utility>

namespace boost
{

namespace range_detail
{

BOOST_RANGE_EXTRACT_OPTIONAL_TYPE( const_iterator )

template< typename C >
struct range_const_iterator_helper
: extract_const_iterator<C>
{};


template< typename Iterator >
struct range_const_iterator_helper<std::pair<Iterator,Iterator> >
{
typedef Iterator type;
};


template< typename T, std::size_t sz >
struct range_const_iterator_helper< T[sz] >
{
typedef const T* type;
};

} 

template<typename C, typename Enabler=void>
struct range_const_iterator
: range_detail::range_const_iterator_helper<
BOOST_DEDUCED_TYPENAME remove_reference<C>::type
>
{
};

} 


#endif
