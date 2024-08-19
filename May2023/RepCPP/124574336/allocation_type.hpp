
#ifndef BOOST_INTERPROCESS_CONTAINERS_ALLOCATION_TYPE_HPP
#define BOOST_INTERPROCESS_CONTAINERS_ALLOCATION_TYPE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/container/detail/allocation_type.hpp>

namespace boost {
namespace interprocess {

#if !defined(BOOST_INTERPROCESS_DOXYGEN_INVOKED)
typedef int allocation_type;
#endif   
static const allocation_type allocate_new       = boost::container::allocate_new;
static const allocation_type expand_fwd         = boost::container::expand_fwd;
static const allocation_type expand_bwd         = boost::container::expand_bwd;
static const allocation_type shrink_in_place    = boost::container::shrink_in_place;
static const allocation_type try_shrink_in_place= boost::container::try_shrink_in_place;
static const allocation_type nothrow_allocation = boost::container::nothrow_allocation;
static const allocation_type zero_memory        = boost::container::zero_memory;

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif 
