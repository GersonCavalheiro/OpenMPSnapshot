
#ifndef BOOST_CONTAINER_PMR_SYNCHRONIZED_POOL_RESOURCE_HPP
#define BOOST_CONTAINER_PMR_SYNCHRONIZED_POOL_RESOURCE_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/detail/auto_link.hpp>
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/detail/pool_resource.hpp>
#include <boost/container/detail/thread_mutex.hpp>

#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

class BOOST_CONTAINER_DECL synchronized_pool_resource
: public memory_resource
{
dtl::thread_mutex m_mut;
pool_resource     m_pool_resource;

public:

synchronized_pool_resource(const pool_options& opts, memory_resource* upstream) BOOST_NOEXCEPT;

synchronized_pool_resource() BOOST_NOEXCEPT;

explicit synchronized_pool_resource(memory_resource* upstream) BOOST_NOEXCEPT;

explicit synchronized_pool_resource(const pool_options& opts) BOOST_NOEXCEPT;

#if !defined(BOOST_NO_CXX11_DELETED_FUNCTIONS) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
synchronized_pool_resource(const synchronized_pool_resource&) = delete;
synchronized_pool_resource operator=(const synchronized_pool_resource&) = delete;
#else
private:
synchronized_pool_resource          (const synchronized_pool_resource&);
synchronized_pool_resource operator=(const synchronized_pool_resource&);
public:
#endif

~synchronized_pool_resource() BOOST_OVERRIDE;

void release();

memory_resource* upstream_resource() const;

pool_options options() const;

protected:

void* do_allocate(std::size_t bytes, std::size_t alignment) BOOST_OVERRIDE;

void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) BOOST_OVERRIDE;

bool do_is_equal(const memory_resource& other) const BOOST_NOEXCEPT BOOST_OVERRIDE;

public:

std::size_t pool_count() const;

std::size_t pool_index(std::size_t bytes) const;

std::size_t pool_next_blocks_per_chunk(std::size_t pool_idx) const;

std::size_t pool_block(std::size_t pool_idx) const;

std::size_t pool_cached_blocks(std::size_t pool_idx) const;
};

}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
