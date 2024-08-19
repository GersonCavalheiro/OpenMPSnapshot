
#ifndef BOOST_CONTAINER_PMR_VECTOR_HPP
#define BOOST_CONTAINER_PMR_VECTOR_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/devector.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>

namespace boost {
namespace container {
namespace pmr {

#if !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)

template <
typename T,
typename GrowthPolicy = growth_factor_60
>
using devector = boost::container::devector<T, GrowthPolicy, polymorphic_allocator<T> >;

#endif

template <
typename T,
typename GrowthPolicy = growth_factor_60
>
struct devector_of
{
typedef boost::container::devector
< T, GrowthPolicy, polymorphic_allocator<T> > type;
};

}  
}  
}  

#endif   
