



#pragma once

namespace hydra_thrust
{
namespace detail
{

template<typename T, typename System>
class temporary_array;

} 
} 

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/tagged_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/contiguous_storage.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/temporary_allocator.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/no_throw_allocator.h>
#include <memory>

namespace hydra_thrust
{
namespace detail
{


template<typename T, typename System>
class temporary_array
: public contiguous_storage<
T,
no_throw_allocator<
temporary_allocator<T,System>
>
>
{
private:
typedef contiguous_storage<
T,
no_throw_allocator<
temporary_allocator<T,System>
>
> super_t;

typedef no_throw_allocator<temporary_allocator<T,System> > alloc_type;

public:
typedef typename super_t::size_type size_type;

__host__ __device__
temporary_array(hydra_thrust::execution_policy<System> &system);

__host__ __device__
temporary_array(hydra_thrust::execution_policy<System> &system, size_type n);

__host__ __device__
temporary_array(int uninit, hydra_thrust::execution_policy<System> &system, size_type n);

template<typename InputIterator>
__host__ __device__
temporary_array(hydra_thrust::execution_policy<System> &system,
InputIterator first,
size_type n);

template<typename InputIterator, typename InputSystem>
__host__ __device__
temporary_array(hydra_thrust::execution_policy<System> &system,
hydra_thrust::execution_policy<InputSystem> &input_system,
InputIterator first,
size_type n);

template<typename InputIterator>
__host__ __device__
temporary_array(hydra_thrust::execution_policy<System> &system,
InputIterator first,
InputIterator last);

template<typename InputSystem, typename InputIterator>
__host__ __device__
temporary_array(hydra_thrust::execution_policy<System> &system,
hydra_thrust::execution_policy<InputSystem> &input_system,
InputIterator first,
InputIterator last);

__host__ __device__
~temporary_array();
}; 


template<typename Iterator, typename System>
class tagged_iterator_range
{
public:
typedef hydra_thrust::detail::tagged_iterator<Iterator,System> iterator;

template<typename Ignored1, typename Ignored2>
tagged_iterator_range(const Ignored1 &, const Ignored2 &, Iterator first, Iterator last)
: m_begin(first),
m_end(last)
{}

iterator begin(void) const { return m_begin; }
iterator end(void) const { return m_end; }

private:
iterator m_begin, m_end;
};


template<typename Iterator, typename FromSystem, typename ToSystem>
struct move_to_system_base
: public eval_if<
is_convertible<
FromSystem,
ToSystem
>::value,
identity_<
tagged_iterator_range<Iterator,ToSystem>
>,
identity_<
temporary_array<
typename hydra_thrust::iterator_value<Iterator>::type,
ToSystem
>
>
>
{};


template<typename Iterator, typename FromSystem, typename ToSystem>
class move_to_system
: public move_to_system_base<
Iterator,
FromSystem,
ToSystem
>::type
{
typedef typename move_to_system_base<Iterator,FromSystem,ToSystem>::type super_t;

public:
move_to_system(hydra_thrust::execution_policy<FromSystem> &from_system,
hydra_thrust::execution_policy<ToSystem> &to_system,
Iterator first,
Iterator last)
: super_t(to_system, from_system, first, last) {}
};


} 
} 

#include <hydra/detail/external/hydra_thrust/detail/temporary_array.inl>

