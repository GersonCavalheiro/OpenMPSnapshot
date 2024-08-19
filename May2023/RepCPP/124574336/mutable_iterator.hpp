
#ifndef BOOST_RANGE_MUTABLE_ITERATOR_HPP
#define BOOST_RANGE_MUTABLE_ITERATOR_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>

#include <boost/range/range_fwd.hpp>
#include <boost/range/detail/extract_optional_type.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <cstddef>
#include <utility>

namespace boost
{


namespace range_detail
{

BOOST_RANGE_EXTRACT_OPTIONAL_TYPE( iterator )

template< typename C >
struct range_mutable_iterator
: range_detail::extract_iterator<
BOOST_DEDUCED_TYPENAME remove_reference<C>::type>
{};


template< typename Iterator >
struct range_mutable_iterator< std::pair<Iterator,Iterator> >
{
typedef Iterator type;
};


template< typename T, std::size_t sz >
struct range_mutable_iterator< T[sz] >
{
typedef T* type;
};

} 

template<typename C, typename Enabler=void>
struct range_mutable_iterator
: range_detail::range_mutable_iterator<
BOOST_DEDUCED_TYPENAME remove_reference<C>::type
>
{
};

} 

#include <boost/range/detail/msvc_has_iterator_workaround.hpp>

#endif
