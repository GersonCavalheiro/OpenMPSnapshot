
#ifndef BOOST_RANGE_TRAVERSAL_HPP
#define BOOST_RANGE_TRAVERSAL_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/iterator.hpp>
#include <boost/iterator/iterator_traits.hpp>

namespace boost
{
template<typename SinglePassRange>
struct range_traversal
: iterator_traversal<typename range_iterator<SinglePassRange>::type>
{
};
}

#endif
