




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <hydra/detail/external/hydra_thrust/allocate_unique.h>
#include <hydra/detail/external/hydra_thrust/device_new.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <hydra/detail/external/hydra_thrust/device_allocator.h>
#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>

HYDRA_THRUST_BEGIN_NS


template <typename T, typename... Args>
__host__
auto device_make_unique(Args&&... args)
-> decltype(
uninitialized_allocate_unique<T>(device_allocator<T>{})
)
{
auto p = uninitialized_allocate_unique<T>(device_allocator<T>{});
device_new<T>(p.get(), T(HYDRA_THRUST_FWD(args)...));
return p;
}


HYDRA_THRUST_END_NS

#endif 
