
#ifndef BOOST_CONTAINER_NEW_ALLOCATOR_HPP
#define BOOST_CONTAINER_NEW_ALLOCATOR_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>
#include <boost/container/throw_exception.hpp>
#include <cstddef>


namespace boost {
namespace container {


template<bool Value>
struct new_allocator_bool
{  static const bool value = Value;  };

template<class T>
class new_allocator;


template<>
class new_allocator<void>
{
public:
typedef void                                 value_type;
typedef void *                               pointer;
typedef const void*                          const_pointer;
typedef BOOST_CONTAINER_IMPDEF(new_allocator_bool<true>) propagate_on_container_move_assignment;
typedef BOOST_CONTAINER_IMPDEF(new_allocator_bool<true>) is_always_equal;

template<class T2>
struct rebind
{
typedef new_allocator< T2> other;
};

new_allocator() BOOST_NOEXCEPT_OR_NOTHROW
{}

new_allocator(const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{}

new_allocator& operator=(const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{
return *this;
}

template<class T2>
new_allocator(const new_allocator<T2> &) BOOST_NOEXCEPT_OR_NOTHROW
{}

friend void swap(new_allocator &, new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{}

friend bool operator==(const new_allocator &, const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return true;   }

friend bool operator!=(const new_allocator &, const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return false;   }
};


template<class T>
class new_allocator
{
public:
typedef T                                    value_type;
typedef T *                                  pointer;
typedef const T *                            const_pointer;
typedef T &                                  reference;
typedef const T &                            const_reference;
typedef std::size_t                          size_type;
typedef std::ptrdiff_t                       difference_type;
typedef BOOST_CONTAINER_IMPDEF(new_allocator_bool<true>) propagate_on_container_move_assignment;
typedef BOOST_CONTAINER_IMPDEF(new_allocator_bool<true>) is_always_equal;

template<class T2>
struct rebind
{
typedef new_allocator<T2> other;
};

new_allocator() BOOST_NOEXCEPT_OR_NOTHROW
{}

new_allocator(const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{}

new_allocator& operator=(const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{
return *this;
}

template<class T2>
new_allocator(const new_allocator<T2> &) BOOST_NOEXCEPT_OR_NOTHROW
{}

pointer allocate(size_type count)
{
const std::size_t max_count = std::size_t(-1)/(2*sizeof(T));
if(BOOST_UNLIKELY(count > max_count))
throw_bad_alloc();
return static_cast<T*>(::operator new(count*sizeof(T)));
}

void deallocate(pointer ptr, size_type) BOOST_NOEXCEPT_OR_NOTHROW
{ ::operator delete((void*)ptr); }

size_type max_size() const BOOST_NOEXCEPT_OR_NOTHROW
{  return std::size_t(-1)/(2*sizeof(T));   }

friend void swap(new_allocator &, new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{}

friend bool operator==(const new_allocator &, const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return true;   }

friend bool operator!=(const new_allocator &, const new_allocator &) BOOST_NOEXCEPT_OR_NOTHROW
{  return false;   }
};

}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
