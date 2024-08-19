

#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#    include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#    include <alpaka/event/EventGenericSycl.hpp>

namespace alpaka
{
using EventFpgaSyclXilinx = EventGenericSycl<DevFpgaSyclXilinx>;
} 

#endif
