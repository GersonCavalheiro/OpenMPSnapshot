
#ifndef BOOST_CONTAINER_PMR_MONOTONIC_BUFFER_RESOURCE_HPP
#define BOOST_CONTAINER_PMR_MONOTONIC_BUFFER_RESOURCE_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/detail/auto_link.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/detail/block_slist.hpp>

#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

class BOOST_CONTAINER_DECL monotonic_buffer_resource
: public memory_resource
{
block_slist       m_memory_blocks;
void *            m_current_buffer;
std::size_t       m_current_buffer_size;
std::size_t       m_next_buffer_size;
void * const      m_initial_buffer;
std::size_t const m_initial_buffer_size;

void increase_next_buffer();
void increase_next_buffer_at_least_to(std::size_t minimum_size);
void *allocate_from_current(std::size_t aligner, std::size_t bytes);

public:

static const std::size_t initial_next_buffer_size = 32u*sizeof(void*);

explicit monotonic_buffer_resource(memory_resource* upstream = 0) BOOST_NOEXCEPT;

explicit monotonic_buffer_resource(std::size_t initial_size, memory_resource* upstream = 0) BOOST_NOEXCEPT;

monotonic_buffer_resource(void* buffer, std::size_t buffer_size, memory_resource* upstream = 0) BOOST_NOEXCEPT;

#if !defined(BOOST_NO_CXX11_DELETED_FUNCTIONS) || defined(BOOST_CONTAINER_DOXYGEN_INVOKED)
monotonic_buffer_resource(const monotonic_buffer_resource&) = delete;
monotonic_buffer_resource operator=(const monotonic_buffer_resource&) = delete;
#else
private:
monotonic_buffer_resource          (const monotonic_buffer_resource&);
monotonic_buffer_resource operator=(const monotonic_buffer_resource&);
public:
#endif

~monotonic_buffer_resource() BOOST_OVERRIDE;

void release() BOOST_NOEXCEPT;

memory_resource* upstream_resource() const BOOST_NOEXCEPT;

std::size_t remaining_storage(std::size_t alignment, std::size_t &wasted_due_to_alignment) const BOOST_NOEXCEPT;

std::size_t remaining_storage(std::size_t alignment = 1u) const BOOST_NOEXCEPT;

const void *current_buffer() const BOOST_NOEXCEPT;

std::size_t next_buffer_size() const BOOST_NOEXCEPT;

protected:

void* do_allocate(std::size_t bytes, std::size_t alignment) BOOST_OVERRIDE;

void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) BOOST_NOEXCEPT BOOST_OVERRIDE;

bool do_is_equal(const memory_resource& other) const BOOST_NOEXCEPT BOOST_OVERRIDE;
};

}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
