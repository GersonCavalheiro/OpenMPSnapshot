
#ifndef BOOST_RANGE_DISTANCE_HPP
#define BOOST_RANGE_DISTANCE_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/iterator/distance.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/difference_type.hpp>

namespace boost
{

namespace range_distance_adl_barrier
{
template< class T >
inline BOOST_CXX14_CONSTEXPR BOOST_DEDUCED_TYPENAME range_difference<T>::type
distance( const T& r )
{
return boost::iterators::distance( boost::begin( r ), boost::end( r ) );
}
}

using namespace range_distance_adl_barrier;

} 

#endif
