
#ifndef BOOST_CONTAINER_PMR_POOL_OPTIONS_HPP
#define BOOST_CONTAINER_PMR_POOL_OPTIONS_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

struct pool_options
{
pool_options()
: max_blocks_per_chunk(0u), largest_required_pool_block(0u)
{}
std::size_t max_blocks_per_chunk;
std::size_t largest_required_pool_block;
};

}  
}  
}  

#endif   
