
#ifndef ASIO_DETAIL_MEMORY_HPP
#define ASIO_DETAIL_MEMORY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include "asio/detail/throw_exception.hpp"

#if !defined(ASIO_HAS_STD_SHARED_PTR)
# include <boost/make_shared.hpp>
# include <boost/shared_ptr.hpp>
# include <boost/weak_ptr.hpp>
#endif 

#if !defined(ASIO_HAS_STD_ADDRESSOF)
# include <boost/utility/addressof.hpp>
#endif 

#if !defined(ASIO_HAS_STD_ALIGNED_ALLOC) \
&& defined(ASIO_HAS_BOOST_ALIGN) \
&& defined(ASIO_HAS_ALIGNOF)
# include <boost/align/aligned_alloc.hpp>
#endif 

namespace asio {
namespace detail {

#if defined(ASIO_HAS_STD_SHARED_PTR)
using std::make_shared;
using std::shared_ptr;
using std::weak_ptr;
#else 
using boost::make_shared;
using boost::shared_ptr;
using boost::weak_ptr;
#endif 

#if defined(ASIO_HAS_STD_ADDRESSOF)
using std::addressof;
#else 
using boost::addressof;
#endif 

} 

#if defined(ASIO_HAS_CXX11_ALLOCATORS)
using std::allocator_arg_t;
# define ASIO_USES_ALLOCATOR(t) \
namespace std { \
template <typename Allocator> \
struct uses_allocator<t, Allocator> : true_type {}; \
} \

# define ASIO_REBIND_ALLOC(alloc, t) \
typename std::allocator_traits<alloc>::template rebind_alloc<t>

#else 
struct allocator_arg_t {};
# define ASIO_USES_ALLOCATOR(t)
# define ASIO_REBIND_ALLOC(alloc, t) \
typename alloc::template rebind<t>::other

#endif 

inline void* aligned_new(std::size_t align, std::size_t size)
{
#if defined(ASIO_HAS_STD_ALIGNED_ALLOC) && defined(ASIO_HAS_ALIGNOF)
size = (size % align == 0) ? size : size + (align - size % align);
void* ptr = std::aligned_alloc(align, size);
if (!ptr)
{
std::bad_alloc ex;
asio::detail::throw_exception(ex);
}
return ptr;
#elif defined(ASIO_HAS_BOOST_ALIGN) && defined(ASIO_HAS_ALIGNOF)
size = (size % align == 0) ? size : size + (align - size % align);
void* ptr = boost::alignment::aligned_alloc(align, size);
if (!ptr)
{
std::bad_alloc ex;
asio::detail::throw_exception(ex);
}
return ptr;
#elif defined(ASIO_MSVC) && defined(ASIO_HAS_ALIGNOF)
size = (size % align == 0) ? size : size + (align - size % align);
void* ptr = _aligned_malloc(size, align);
if (!ptr)
{
std::bad_alloc ex;
asio::detail::throw_exception(ex);
}
return ptr;
#else 
(void)align;
return ::operator new(size);
#endif 
}

inline void aligned_delete(void* ptr)
{
#if defined(ASIO_HAS_STD_ALIGNED_ALLOC) && defined(ASIO_HAS_ALIGNOF)
std::free(ptr);
#elif defined(ASIO_HAS_BOOST_ALIGN) && defined(ASIO_HAS_ALIGNOF)
boost::alignment::aligned_free(ptr);
#elif defined(ASIO_MSVC) && defined(ASIO_HAS_ALIGNOF)
_aligned_free(ptr);
#else 
::operator delete(ptr);
#endif 
}

} 

#endif 
