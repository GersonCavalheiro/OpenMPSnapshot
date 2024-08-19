

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/detail/normal_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/allocator_traits.h>

namespace hydra_thrust
{

namespace detail
{

struct copy_allocator_t {};

template<typename T, typename Alloc>
class contiguous_storage
{
private:
typedef hydra_thrust::detail::allocator_traits<Alloc> alloc_traits;

public:
typedef Alloc                                      allocator_type;
typedef T                                          value_type;
typedef typename alloc_traits::pointer             pointer;
typedef typename alloc_traits::const_pointer       const_pointer;
typedef typename alloc_traits::size_type           size_type;
typedef typename alloc_traits::difference_type     difference_type;

typedef typename Alloc::reference                  reference;
typedef typename Alloc::const_reference            const_reference;

typedef hydra_thrust::detail::normal_iterator<pointer>       iterator;
typedef hydra_thrust::detail::normal_iterator<const_pointer> const_iterator;

__hydra_thrust_exec_check_disable__
__host__ __device__
explicit contiguous_storage(const allocator_type &alloc = allocator_type());

__hydra_thrust_exec_check_disable__
__host__ __device__
explicit contiguous_storage(size_type n, const allocator_type &alloc = allocator_type());

__hydra_thrust_exec_check_disable__
__host__ __device__
explicit contiguous_storage(copy_allocator_t, const contiguous_storage &other);

__hydra_thrust_exec_check_disable__
__host__ __device__
explicit contiguous_storage(copy_allocator_t, const contiguous_storage &other, size_type n);

__hydra_thrust_exec_check_disable__
__host__ __device__
~contiguous_storage();

__host__ __device__
size_type size() const;

__host__ __device__
size_type max_size() const;

__host__ __device__
pointer data();

__host__ __device__
const_pointer data() const;

__host__ __device__
iterator begin();

__host__ __device__
const_iterator begin() const;

__host__ __device__
iterator end();

__host__ __device__
const_iterator end() const;

__host__ __device__
reference operator[](size_type n);

__host__ __device__
const_reference operator[](size_type n) const;

__host__ __device__
allocator_type get_allocator() const;

__host__ __device__
void allocate(size_type n);

__host__ __device__
void deallocate();

__host__ __device__
void swap(contiguous_storage &x);

__host__ __device__
void default_construct_n(iterator first, size_type n);

__host__ __device__
void uninitialized_fill_n(iterator first, size_type n, const value_type &value);

template<typename InputIterator>
__host__ __device__
iterator uninitialized_copy(InputIterator first, InputIterator last, iterator result);

template<typename System, typename InputIterator>
__host__ __device__
iterator uninitialized_copy(hydra_thrust::execution_policy<System> &from_system,
InputIterator first,
InputIterator last,
iterator result);

template<typename InputIterator, typename Size>
__host__ __device__
iterator uninitialized_copy_n(InputIterator first, Size n, iterator result);

template<typename System, typename InputIterator, typename Size>
__host__ __device__
iterator uninitialized_copy_n(hydra_thrust::execution_policy<System> &from_system,
InputIterator first,
Size n,
iterator result);

__host__ __device__
void destroy(iterator first, iterator last);

__host__ __device__
void deallocate_on_allocator_mismatch(const contiguous_storage &other);

__host__ __device__
void destroy_on_allocator_mismatch(const contiguous_storage &other,
iterator first, iterator last);

__host__ __device__
void set_allocator(const allocator_type &alloc);

__host__ __device__
bool is_allocator_not_equal(const allocator_type &alloc) const;

__host__ __device__
bool is_allocator_not_equal(const contiguous_storage &other) const;

__host__ __device__
void propagate_allocator(const contiguous_storage &other);

#if __cplusplus >= 201103L
__host__ __device__
void propagate_allocator(contiguous_storage &other);

__host__ __device__
contiguous_storage &operator=(contiguous_storage &&other);
#endif

private:
allocator_type m_allocator;

iterator m_begin;

size_type m_size;

contiguous_storage &operator=(const contiguous_storage &x);

__host__ __device__
void swap_allocators(true_type, const allocator_type &);

__host__ __device__
void swap_allocators(false_type, allocator_type &);

__host__ __device__
bool is_allocator_not_equal_dispatch(true_type, const allocator_type &) const;

__host__ __device__
bool is_allocator_not_equal_dispatch(false_type, const allocator_type &) const;

__host__ __device__
void deallocate_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other);

__host__ __device__
void deallocate_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &other);

__host__ __device__
void destroy_on_allocator_mismatch_dispatch(true_type, const contiguous_storage &other,
iterator first, iterator last);

__host__ __device__
void destroy_on_allocator_mismatch_dispatch(false_type, const contiguous_storage &other,
iterator first, iterator last);

__host__ __device__
void propagate_allocator_dispatch(true_type, const contiguous_storage &other);

__host__ __device__
void propagate_allocator_dispatch(false_type, const contiguous_storage &other);

#if __cplusplus >= 201103L
__host__ __device__
void propagate_allocator_dispatch(true_type, contiguous_storage &other);

__host__ __device__
void propagate_allocator_dispatch(false_type, contiguous_storage &other);
#endif
}; 

} 

template<typename T, typename Alloc>
__host__ __device__
void swap(detail::contiguous_storage<T,Alloc> &lhs, detail::contiguous_storage<T,Alloc> &rhs);

} 

#include <hydra/detail/external/hydra_thrust/detail/contiguous_storage.inl>

