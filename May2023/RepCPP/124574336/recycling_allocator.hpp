
#ifndef BOOST_ASIO_DETAIL_RECYCLING_ALLOCATOR_HPP
#define BOOST_ASIO_DETAIL_RECYCLING_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/thread_context.hpp>
#include <boost/asio/detail/thread_info_base.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T, typename Purpose = thread_info_base::default_tag>
class recycling_allocator
{
public:
typedef T value_type;

template <typename U>
struct rebind
{
typedef recycling_allocator<U, Purpose> other;
};

recycling_allocator()
{
}

template <typename U>
recycling_allocator(const recycling_allocator<U, Purpose>&)
{
}

T* allocate(std::size_t n)
{
typedef thread_context::thread_call_stack call_stack;
void* p = thread_info_base::allocate(Purpose(),
call_stack::top(), sizeof(T) * n);
return static_cast<T*>(p);
}

void deallocate(T* p, std::size_t n)
{
typedef thread_context::thread_call_stack call_stack;
thread_info_base::deallocate(Purpose(),
call_stack::top(), p, sizeof(T) * n);
}
};

template <typename Purpose>
class recycling_allocator<void, Purpose>
{
public:
typedef void value_type;

template <typename U>
struct rebind
{
typedef recycling_allocator<U, Purpose> other;
};

recycling_allocator()
{
}

template <typename U>
recycling_allocator(const recycling_allocator<U, Purpose>&)
{
}
};

template <typename Allocator, typename Purpose>
struct get_recycling_allocator
{
typedef Allocator type;
static type get(const Allocator& a) { return a; }
};

template <typename T, typename Purpose>
struct get_recycling_allocator<std::allocator<T>, Purpose>
{
typedef recycling_allocator<T, Purpose> type;
static type get(const std::allocator<T>&) { return type(); }
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
