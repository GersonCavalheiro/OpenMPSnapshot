

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>

namespace alpaka
{
using QueueFpgaSyclXilinxNonBlocking = QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>;
}

#endif