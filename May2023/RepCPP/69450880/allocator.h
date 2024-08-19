



#pragma once

#include <limits>

#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>

#include <hydra/detail/external/hydra_thrust/mr/detail/config.h>
#include <hydra/detail/external/hydra_thrust/mr/validator.h>
#include <hydra/detail/external/hydra_thrust/mr/polymorphic_adaptor.h>

namespace hydra_thrust
{
namespace mr
{




template<typename T, class MR>
class allocator : private validator<MR>
{
public:

typedef typename MR::pointer void_pointer;


typedef T value_type;

typedef typename hydra_thrust::detail::pointer_traits<void_pointer>::template rebind<T>::other pointer;

typedef typename hydra_thrust::detail::pointer_traits<void_pointer>::template rebind<const T>::other const_pointer;

typedef typename hydra_thrust::detail::pointer_traits<pointer>::reference reference;

typedef typename hydra_thrust::detail::pointer_traits<const_pointer>::reference const_reference;

typedef std::size_t size_type;

typedef typename hydra_thrust::detail::pointer_traits<pointer>::difference_type difference_type;


typedef detail::true_type propagate_on_container_copy_assignment;

typedef detail::true_type propagate_on_container_move_assignment;

typedef detail::true_type propagate_on_container_swap;


template<typename U>
struct rebind
{

typedef allocator<U, MR> other;
};


__host__ __device__
size_type max_size() const
{
return std::numeric_limits<size_type>::max() / sizeof(T);
}


__host__ __device__
allocator(MR * resource) : mem_res(resource)
{
}


template<typename U>
__host__ __device__
allocator(const allocator<U, MR> & other) : mem_res(other.resource())
{
}


HYDRA_THRUST_NODISCARD
__host__
pointer allocate(size_type n)
{
return static_cast<pointer>(mem_res->do_allocate(n * sizeof(T), HYDRA_THRUST_ALIGNOF(T)));
}


__host__
void deallocate(pointer p, size_type n)
{
return mem_res->do_deallocate(p, n * sizeof(T), HYDRA_THRUST_ALIGNOF(T));
}


__host__ __device__
MR * resource() const
{
return mem_res;
}

private:
MR * mem_res;
};


template<typename T, typename MR>
__host__ __device__
bool operator==(const allocator<T, MR> & lhs, const allocator<T, MR> & rhs) HYDRA_THRUST_NOEXCEPT
{
return *lhs.resource() == *rhs.resource();
}


template<typename T, typename MR>
__host__ __device__
bool operator!=(const allocator<T, MR> & lhs, const allocator<T, MR> & rhs) HYDRA_THRUST_NOEXCEPT
{
return !(lhs == rhs);
}

#if __cplusplus >= 201103L

template<typename T, typename Pointer>
using polymorphic_allocator = allocator<T, polymorphic_adaptor_resource<Pointer> >;

#else

template<typename T, typename Pointer>
class polymorphic_allocator : public allocator<T, polymorphic_adaptor_resource<Pointer> >
{
typedef allocator<T, polymorphic_adaptor_resource<Pointer> > base;

public:

polymorphic_allocator(polymorphic_adaptor_resource<Pointer>  * resource) : base(resource)
{
}
};

#endif


template<typename T, typename Upstream>
class stateless_resource_allocator : public hydra_thrust::mr::allocator<T, Upstream>
{
typedef hydra_thrust::mr::allocator<T, Upstream> base;

public:

template<typename U>
struct rebind
{

typedef stateless_resource_allocator<U, Upstream> other;
};


__host__
stateless_resource_allocator() : base(get_global_resource<Upstream>())
{
}


__host__ __device__
stateless_resource_allocator(const stateless_resource_allocator & other)
: base(other) {}


template<typename U>
__host__ __device__
stateless_resource_allocator(const stateless_resource_allocator<U, Upstream> & other)
: base(other) {}


__host__ __device__
~stateless_resource_allocator() {}
};

} 
} 

