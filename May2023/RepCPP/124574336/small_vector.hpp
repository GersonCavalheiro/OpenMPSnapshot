
#ifndef BOOST_CONTAINER_PMR_SMALL_VECTOR_HPP
#define BOOST_CONTAINER_PMR_SMALL_VECTOR_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/small_vector.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>

namespace boost {
namespace container {
namespace pmr {

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)

template <class T, std::size_t N>
using small_vector = boost::container::small_vector<T, N, polymorphic_allocator<T>>;

#endif

template<class T, std::size_t N>
struct small_vector_of
{
typedef boost::container::small_vector
< T, N, polymorphic_allocator<T> > type;
};

}  
}  
}  

#endif   
