

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/contiguous_storage.h>
#include <hydra/detail/external/hydra_thrust/detail/swap.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/allocator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/copy_construct_range.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/default_construct_range.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/destroy_range.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/fill_construct_range.h>
#include <utility> 

namespace hydra_thrust
{

namespace detail
{

class allocator_mismatch_on_swap : public std::runtime_error
{
public:
allocator_mismatch_on_swap()
:std::runtime_error("swap called on containers with allocators that propagate on swap, but compare non-equal")
{
}
};

__hydra_thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
contiguous_storage<T,Alloc>
::contiguous_storage(const Alloc &alloc)
:m_allocator(alloc),
m_begin(pointer(static_cast<T*>(0))),
m_size(0)
{
;
} 

__hydra_thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
contiguous_storage<T,Alloc>
::contiguous_storage(size_type n, const Alloc &alloc)
:m_allocator(alloc),
m_begin(pointer(static_cast<T*>(0))),
m_size(0)
{
allocate(n);
} 

template<typename T, typename Alloc>
__host__ __device__
contiguous_storage<T,Alloc>
::contiguous_storage(copy_allocator_t,
const contiguous_storage &other)
:m_allocator(other.m_allocator),
m_begin(pointer(static_cast<T*>(0))),
m_size(0)
{
} 

template<typename T, typename Alloc>
__host__ __device__
contiguous_storage<T,Alloc>
::contiguous_storage(copy_allocator_t,
const contiguous_storage &other, size_type n)
:m_allocator(other.m_allocator),
m_begin(pointer(static_cast<T*>(0))),
m_size(0)
{
allocate(n);
} 

__hydra_thrust_exec_check_disable__
template<typename T, typename Alloc>
__host__ __device__
contiguous_storage<T,Alloc>
::~contiguous_storage()
{
deallocate();
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::size_type
contiguous_storage<T,Alloc>
::size() const
{
return m_size;
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::size_type
contiguous_storage<T,Alloc>
::max_size() const
{
return alloc_traits::max_size(m_allocator);
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::iterator
contiguous_storage<T,Alloc>
::begin()
{
return m_begin;
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::const_iterator
contiguous_storage<T,Alloc>
::begin() const
{
return m_begin;
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::iterator
contiguous_storage<T,Alloc>
::end()
{
return m_begin + size();
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::const_iterator
contiguous_storage<T,Alloc>
::end() const
{
return m_begin + size();
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::pointer
contiguous_storage<T,Alloc>
::data()
{
return &*m_begin;
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::const_pointer
contiguous_storage<T,Alloc>
::data() const
{
return &*m_begin;
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::reference
contiguous_storage<T,Alloc>
::operator[](size_type n)
{
return m_begin[n];
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::const_reference
contiguous_storage<T,Alloc>
::operator[](size_type n) const
{
return m_begin[n];
} 

template<typename T, typename Alloc>
__host__ __device__
typename contiguous_storage<T,Alloc>::allocator_type
contiguous_storage<T,Alloc>
::get_allocator() const
{
return m_allocator;
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::allocate(size_type n)
{
if(n > 0)
{
m_begin = iterator(alloc_traits::allocate(m_allocator,n));
m_size = n;
} 
else
{
m_begin = iterator(pointer(static_cast<T*>(0)));
m_size = 0;
} 
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::deallocate()
{
if(size() > 0)
{
alloc_traits::deallocate(m_allocator,m_begin.base(), size());
m_begin = iterator(pointer(static_cast<T*>(0)));
m_size = 0;
} 
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::swap(contiguous_storage &x)
{
hydra_thrust::swap(m_begin, x.m_begin);
hydra_thrust::swap(m_size, x.m_size);

swap_allocators(
integral_constant<
bool,
allocator_traits<Alloc>::propagate_on_container_swap::value
>(),
x.m_allocator);

hydra_thrust::swap(m_allocator, x.m_allocator);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::default_construct_n(iterator first, size_type n)
{
default_construct_range(m_allocator, first.base(), n);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::uninitialized_fill_n(iterator first, size_type n, const value_type &x)
{
fill_construct_range(m_allocator, first.base(), n, x);
} 

template<typename T, typename Alloc>
template<typename System, typename InputIterator>
__host__ __device__
typename contiguous_storage<T,Alloc>::iterator
contiguous_storage<T,Alloc>
::uninitialized_copy(hydra_thrust::execution_policy<System> &from_system, InputIterator first, InputIterator last, iterator result)
{
return iterator(copy_construct_range(from_system, m_allocator, first, last, result.base()));
} 

template<typename T, typename Alloc>
template<typename InputIterator>
__host__ __device__
typename contiguous_storage<T,Alloc>::iterator
contiguous_storage<T,Alloc>
::uninitialized_copy(InputIterator first, InputIterator last, iterator result)
{
typename hydra_thrust::iterator_system<InputIterator>::type from_system;

return iterator(copy_construct_range(from_system, m_allocator, first, last, result.base()));
} 

template<typename T, typename Alloc>
template<typename System, typename InputIterator, typename Size>
__host__ __device__
typename contiguous_storage<T,Alloc>::iterator
contiguous_storage<T,Alloc>
::uninitialized_copy_n(hydra_thrust::execution_policy<System> &from_system, InputIterator first, Size n, iterator result)
{
return iterator(copy_construct_range_n(from_system, m_allocator, first, n, result.base()));
} 

template<typename T, typename Alloc>
template<typename InputIterator, typename Size>
__host__ __device__
typename contiguous_storage<T,Alloc>::iterator
contiguous_storage<T,Alloc>
::uninitialized_copy_n(InputIterator first, Size n, iterator result)
{
typename hydra_thrust::iterator_system<InputIterator>::type from_system;

return iterator(copy_construct_range_n(from_system, m_allocator, first, n, result.base()));
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::destroy(iterator first, iterator last)
{
destroy_range(m_allocator, first.base(), last - first);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::deallocate_on_allocator_mismatch(const contiguous_storage &other)
{
integral_constant<
bool,
allocator_traits<Alloc>::propagate_on_container_copy_assignment::value
> c;

deallocate_on_allocator_mismatch_dispatch(c, other);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::destroy_on_allocator_mismatch(const contiguous_storage &other,
iterator first, iterator last)
{
integral_constant<
bool,
allocator_traits<Alloc>::propagate_on_container_copy_assignment::value
> c;

destroy_on_allocator_mismatch_dispatch(c, other, first, last);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::set_allocator(const Alloc &alloc)
{
m_allocator = alloc;
} 

template<typename T, typename Alloc>
__host__ __device__
bool contiguous_storage<T,Alloc>
::is_allocator_not_equal(const Alloc &alloc) const
{
return is_allocator_not_equal_dispatch(
integral_constant<
bool,
allocator_traits<Alloc>::is_always_equal::value
>(),
alloc);
} 

template<typename T, typename Alloc>
__host__ __device__
bool contiguous_storage<T,Alloc>
::is_allocator_not_equal(const contiguous_storage<T,Alloc> &other) const
{
return is_allocator_not_equal(m_allocator, other.m_allocator);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::propagate_allocator(const contiguous_storage &other)
{
integral_constant<
bool,
allocator_traits<Alloc>::propagate_on_container_copy_assignment::value
> c;

propagate_allocator_dispatch(c, other);
} 

#if __cplusplus >= 201103L
template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::propagate_allocator(contiguous_storage &other)
{
integral_constant<
bool,
allocator_traits<Alloc>::propagate_on_container_move_assignment::value
> c;

propagate_allocator_dispatch(c, other);
} 

template<typename T, typename Alloc>
__host__ __device__
contiguous_storage<T,Alloc> &contiguous_storage<T,Alloc>
::operator=(contiguous_storage &&other)
{
if (size() > 0)
{
deallocate();
}
propagate_allocator(other);
m_begin = std::move(other.m_begin);
m_size = std::move(other.m_size);

other.m_begin = pointer(static_cast<T*>(0));
other.m_size = 0;

return *this;
} 
#endif

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::swap_allocators(true_type, const Alloc &)
{
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::swap_allocators(false_type, Alloc &other)
{
#ifdef __CUDA_ARCH__
assert(!is_allocator_not_equal(other));
#else
if (is_allocator_not_equal(other))
{
throw allocator_mismatch_on_swap();
}
#endif
hydra_thrust::swap(m_allocator, other);
} 

template<typename T, typename Alloc>
__host__ __device__
bool contiguous_storage<T,Alloc>
::is_allocator_not_equal_dispatch(true_type , const Alloc &) const
{
return false;
} 

template<typename T, typename Alloc>
__host__ __device__
bool contiguous_storage<T,Alloc>
::is_allocator_not_equal_dispatch(false_type , const Alloc& other) const
{
return m_allocator != other;
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::deallocate_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other)
{
if (m_allocator != other.m_allocator)
{
deallocate();
}
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::deallocate_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &)
{
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::destroy_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other,
iterator first, iterator last)
{
if (m_allocator != other.m_allocator)
{
destroy(first, last);
}
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::destroy_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &,
iterator, iterator)
{
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::propagate_allocator_dispatch(true_type, const contiguous_storage &other)
{
m_allocator = other.m_allocator;
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::propagate_allocator_dispatch(false_type, const contiguous_storage &)
{
} 

#if __cplusplus >= 201103L
template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::propagate_allocator_dispatch(true_type, contiguous_storage &other)
{
m_allocator = std::move(other.m_allocator);
} 

template<typename T, typename Alloc>
__host__ __device__
void contiguous_storage<T,Alloc>
::propagate_allocator_dispatch(false_type, contiguous_storage &)
{
} 
#endif

} 

template<typename T, typename Alloc>
__host__ __device__
void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs)
{
lhs.swap(rhs);
} 

} 

