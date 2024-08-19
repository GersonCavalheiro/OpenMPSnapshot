

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/queue/sycl/QueueGenericSyclBase.hpp>

#    include <memory>
#    include <utility>

namespace alpaka
{
template<typename TDev>
using QueueGenericSyclBlocking = detail::QueueGenericSyclBase<TDev, true>;
} 

#endif
