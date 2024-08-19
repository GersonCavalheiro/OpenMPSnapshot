
#pragma once


#include <hydra/detail/external/hydra_thrust/system/cuda/config.h>

#include <hydra/detail/external/hydra_thrust/distance.h>
#include <hydra/detail/external/hydra_thrust/advance.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/uninitialized_copy.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/util.h>
#include <hydra/detail/external/hydra_thrust/detail/temporary_array.h>
#include <hydra/detail/external/hydra_thrust/type_traits/is_trivially_relocatable.h>

HYDRA_THRUST_BEGIN_NS
namespace cuda_cub {

namespace __copy {


template <class H,
class D,
class T,
class Size>
HYDRA_THRUST_HOST_FUNCTION void
trivial_device_copy(hydra_thrust::cpp::execution_policy<H>&      ,
hydra_thrust::cuda_cub::execution_policy<D>& device_s,
T*                                     dst,
T const*                               src,
Size                                   count)
{
cudaError status;
status = cuda_cub::trivial_copy_to_device(dst,
src,
count,
cuda_cub::stream(device_s));
cuda_cub::throw_on_error(status, "__copy::trivial_device_copy H->D: failed");
}

template <class D,
class H,
class T,
class Size>
HYDRA_THRUST_HOST_FUNCTION void
trivial_device_copy(hydra_thrust::cuda_cub::execution_policy<D>& device_s,
hydra_thrust::cpp::execution_policy<H>&      ,
T*                                     dst,
T const*                               src,
Size                                   count)
{
cudaError status;
status = cuda_cub::trivial_copy_from_device(dst,
src,
count,
cuda_cub::stream(device_s));
cuda_cub::throw_on_error(status, "trivial_device_copy D->H failed");
}

template <class System1,
class System2,
class InputIt,
class Size,
class OutputIt>
OutputIt __host__
cross_system_copy_n(hydra_thrust::execution_policy<System1>& sys1,
hydra_thrust::execution_policy<System2>& sys2,
InputIt                            begin,
Size                               n,
OutputIt                           result,
hydra_thrust::detail::true_type)    

{
typedef typename iterator_traits<InputIt>::value_type InputTy;

trivial_device_copy(derived_cast(sys1),
derived_cast(sys2),
reinterpret_cast<InputTy*>(hydra_thrust::raw_pointer_cast(&*result)),
reinterpret_cast<InputTy const*>(hydra_thrust::raw_pointer_cast(&*begin)),
n);

return result + n;
}

template <class H,
class D,
class InputIt,
class Size,
class OutputIt>
OutputIt __host__
cross_system_copy_n(hydra_thrust::cpp::execution_policy<H>&      host_s,
hydra_thrust::cuda_cub::execution_policy<D>& device_s,
InputIt                                first,
Size                                   num_items,
OutputIt                               result,
hydra_thrust::detail::false_type)    
{
typedef typename hydra_thrust::iterator_value<InputIt>::type InputTy;

InputIt last = first;
hydra_thrust::advance(last, num_items);
hydra_thrust::detail::temporary_array<InputTy, H> temp(host_s, num_items);

for (Size idx = 0; idx != num_items; idx++)
{
::new (static_cast<void*>(temp.data().get()+idx)) InputTy(*first);
++first;
}

hydra_thrust::detail::temporary_array<InputTy, D> d_in_ptr(device_s, num_items);

cudaError status = cuda_cub::trivial_copy_to_device(d_in_ptr.data().get(),
temp.data().get(),
num_items,
cuda_cub::stream(device_s));
cuda_cub::throw_on_error(status, "__copy:: H->D: failed");


OutputIt ret = cuda_cub::copy_n(device_s, d_in_ptr.data(), num_items, result);

return ret;
}

#if HYDRA_THRUST_DEVICE_COMPILER == HYDRA_THRUST_DEVICE_COMPILER_NVCC
template <class D,
class H,
class InputIt,
class Size,
class OutputIt>
OutputIt __host__
cross_system_copy_n(hydra_thrust::cuda_cub::execution_policy<D>& device_s,
hydra_thrust::cpp::execution_policy<H>&   host_s,
InputIt                             first,
Size                                num_items,
OutputIt                            result,
hydra_thrust::detail::false_type)    

{
typedef typename hydra_thrust::iterator_value<InputIt>::type InputTy;

hydra_thrust::detail::temporary_array<InputTy, D> d_in_ptr(device_s, num_items);

cuda_cub::uninitialized_copy_n(device_s, first, num_items, d_in_ptr.data());

hydra_thrust::detail::temporary_array<InputTy, H> temp(host_s, num_items);

cudaError status;
status = cuda_cub::trivial_copy_from_device(temp.data().get(),
d_in_ptr.data().get(),
num_items,
cuda_cub::stream(device_s));
cuda_cub::throw_on_error(status, "__copy:: D->H: failed");

OutputIt ret = hydra_thrust::copy_n(host_s, temp.data(), num_items, result);

return ret;
}
#endif

template <class System1,
class System2,
class InputIt,
class Size,
class OutputIt>
OutputIt __host__
cross_system_copy_n(cross_system<System1, System2> systems,
InputIt  begin,
Size     n,
OutputIt result)
{
return cross_system_copy_n(
derived_cast(systems.sys1),
derived_cast(systems.sys2),
begin,
n,
result,
typename is_indirectly_trivially_relocatable_to<InputIt, OutputIt>::type());
}

template <class System1,
class System2,
class InputIterator,
class OutputIterator>
OutputIterator __host__
cross_system_copy(cross_system<System1, System2> systems,
InputIterator  begin,
InputIterator  end,
OutputIterator result)
{
return cross_system_copy_n(systems,
begin,
hydra_thrust::distance(begin, end),
result);
}

}    

} 
HYDRA_THRUST_END_NS
