
#ifndef BOOST_CONTAINER_PMR_GLOBAL_RESOURCE_HPP
#define BOOST_CONTAINER_PMR_GLOBAL_RESOURCE_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/detail/auto_link.hpp>
#include <boost/container/container_fwd.hpp>

#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

BOOST_CONTAINER_DECL memory_resource* new_delete_resource() BOOST_NOEXCEPT;

BOOST_CONTAINER_DECL memory_resource* null_memory_resource() BOOST_NOEXCEPT;

BOOST_CONTAINER_DECL memory_resource* set_default_resource(memory_resource* r) BOOST_NOEXCEPT;

BOOST_CONTAINER_DECL memory_resource* get_default_resource() BOOST_NOEXCEPT;

}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
