
#ifndef BOOST_CONTAINER_POOL_RESOURCE_HPP
#define BOOST_CONTAINER_POOL_RESOURCE_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/detail/block_list.hpp>
#include <boost/container/pmr/pool_options.hpp>

#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

#if !defined(BOOST_CONTAINER_DOXYGEN_INVOKED)

class pool_data_t;

static const std::size_t pool_options_minimum_max_blocks_per_chunk = 1u;
static const std::size_t pool_options_default_max_blocks_per_chunk = 32u;
static const std::size_t pool_options_minimum_largest_required_pool_block =
memory_resource::max_align > 2*sizeof(void*) ? memory_resource::max_align : 2*sizeof(void*);
static const std::size_t pool_options_default_largest_required_pool_block =
pool_options_minimum_largest_required_pool_block > 4096u
? pool_options_minimum_largest_required_pool_block : 4096u;

#endif   

class pool_resource
{
typedef block_list_base<> block_list_base_t;

pool_options m_options;
memory_resource&   m_upstream;
block_list_base_t  m_oversized_list;
pool_data_t *m_pool_data;
std::size_t  m_pool_count;

static void priv_limit_option(std::size_t &val, std::size_t min, std::size_t max);
static std::size_t priv_pool_index(std::size_t block_size);
static std::size_t priv_pool_block(std::size_t index);

void priv_fix_options();
void priv_init_pools();
void priv_constructor_body();

public:

pool_resource(const pool_options& opts, memory_resource* upstream) BOOST_NOEXCEPT;

pool_resource() BOOST_NOEXCEPT;

explicit pool_resource(memory_resource* upstream) BOOST_NOEXCEPT;

explicit pool_resource(const pool_options& opts) BOOST_NOEXCEPT;

#if !defined(BOOST_NO_CXX11_DELETED_FUNCTIONS) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
pool_resource(const pool_resource&) = delete;
pool_resource operator=(const pool_resource&) = delete;
#else
private:
pool_resource          (const pool_resource&);
pool_resource operator=(const pool_resource&);
public:
#endif

virtual ~pool_resource();

void release();

memory_resource* upstream_resource() const;

pool_options options() const;

public:  

virtual void* do_allocate(std::size_t bytes, std::size_t alignment);

virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment);

virtual bool do_is_equal(const memory_resource& other) const BOOST_NOEXCEPT;

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
