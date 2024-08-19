



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/mr/new.h>
#include <hydra/detail/external/hydra_thrust/mr/fancy_pointer_resource.h>

#include <hydra/detail/external/hydra_thrust/system/tbb/pointer.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{

namespace detail
{
typedef hydra_thrust::mr::fancy_pointer_resource<
hydra_thrust::mr::new_delete_resource,
hydra_thrust::tbb::pointer<void>
> native_resource;
}




typedef detail::native_resource memory_resource;

typedef detail::native_resource universal_memory_resource;

typedef detail::native_resource universal_host_pinned_memory_resource;



}
}
}
