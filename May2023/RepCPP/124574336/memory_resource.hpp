
#ifndef BOOST_CONTAINER_PMR_MEMORY_RESOURCE_HPP
#define BOOST_CONTAINER_PMR_MEMORY_RESOURCE_HPP

#if defined (_MSC_VER)
#  pragma once 
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/container_fwd.hpp>
#include <boost/move/detail/type_traits.hpp>
#include <cstddef>

namespace boost {
namespace container {
namespace pmr {

class memory_resource
{
public:
static BOOST_CONSTEXPR_OR_CONST std::size_t max_align =
boost::move_detail::alignment_of<boost::move_detail::max_align_t>::value;

virtual ~memory_resource(){}

void* allocate(std::size_t bytes, std::size_t alignment = max_align)
{  return this->do_allocate(bytes, alignment);  }

void  deallocate(void* p, std::size_t bytes, std::size_t alignment = max_align)
{  return this->do_deallocate(p, bytes, alignment);  }

bool is_equal(const memory_resource& other) const BOOST_NOEXCEPT
{  return this->do_is_equal(other);  }

#if !defined(BOOST_EMBTC)

friend bool operator==(const memory_resource& a, const memory_resource& b) BOOST_NOEXCEPT
{  return &a == &b || a.is_equal(b);   }

friend bool operator!=(const memory_resource& a, const memory_resource& b) BOOST_NOEXCEPT
{  return !(a == b); }

#else

friend bool operator==(const memory_resource& a, const memory_resource& b) BOOST_NOEXCEPT;

friend bool operator!=(const memory_resource& a, const memory_resource& b) BOOST_NOEXCEPT;

#endif

protected:
virtual void* do_allocate(std::size_t bytes, std::size_t alignment) = 0;

virtual void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) = 0;

virtual bool do_is_equal(const memory_resource& other) const BOOST_NOEXCEPT = 0;
};

#if defined(BOOST_EMBTC)

inline bool operator==(const memory_resource& a, const memory_resource& b) BOOST_NOEXCEPT
{  return &a == &b || a.is_equal(b);   }

inline bool operator!=(const memory_resource& a, const memory_resource& b) BOOST_NOEXCEPT
{  return !(a == b); }

#endif

}  
}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
