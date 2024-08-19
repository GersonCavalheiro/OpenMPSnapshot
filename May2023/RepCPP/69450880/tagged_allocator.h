

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{
namespace detail
{

template<typename T, typename Tag, typename Pointer> class tagged_allocator;

template<typename Tag, typename Pointer>
class tagged_allocator<void, Tag, Pointer>
{
public:
typedef void                                                                                 value_type;
typedef typename hydra_thrust::detail::pointer_traits<Pointer>::template rebind<void>::other       pointer;
typedef typename hydra_thrust::detail::pointer_traits<Pointer>::template rebind<const void>::other const_pointer;
typedef std::size_t                                                                          size_type;
typedef typename hydra_thrust::detail::pointer_traits<Pointer>::difference_type                    difference_type;
typedef Tag                                                                                  system_type;

template<typename U>
struct rebind
{
typedef tagged_allocator<U,Tag,Pointer> other;
}; 
};

template<typename T, typename Tag, typename Pointer>
class tagged_allocator
{
public:
typedef T                                                                                 value_type;
typedef typename hydra_thrust::detail::pointer_traits<Pointer>::template rebind<T>::other       pointer;
typedef typename hydra_thrust::detail::pointer_traits<Pointer>::template rebind<const T>::other const_pointer;
typedef typename hydra_thrust::iterator_reference<pointer>::type                                reference;
typedef typename hydra_thrust::iterator_reference<const_pointer>::type                          const_reference;
typedef std::size_t                                                                       size_type;
typedef typename hydra_thrust::detail::pointer_traits<pointer>::difference_type                 difference_type;
typedef Tag                                                                               system_type;

template<typename U>
struct rebind
{
typedef tagged_allocator<U,Tag,Pointer> other;
}; 

__host__ __device__
inline tagged_allocator();

__host__ __device__
inline tagged_allocator(const tagged_allocator &);

template<typename U, typename OtherPointer>
__host__ __device__
inline tagged_allocator(const tagged_allocator<U, Tag, OtherPointer> &);

__host__ __device__
inline ~tagged_allocator();

__host__ __device__
pointer address(reference x) const;

__host__ __device__
const_pointer address(const_reference x) const;

size_type max_size() const;
};

template<typename T1, typename Pointer1, typename T2, typename Pointer2, typename Tag>
__host__ __device__
bool operator==(const tagged_allocator<T1,Pointer1,Tag> &, const tagged_allocator<T2,Pointer2,Tag> &);

template<typename T1, typename Pointer1, typename T2, typename Pointer2, typename Tag>
__host__ __device__
bool operator!=(const tagged_allocator<T1,Pointer1,Tag> &, const tagged_allocator<T2,Pointer2,Tag> &);

} 
} 

#include <hydra/detail/external/hydra_thrust/detail/allocator/tagged_allocator.inl>

