

#pragma once

#include <alpaka/alpaka.hpp>

#include <condition_variable>
#include <mutex>
#include <utility>

namespace alpaka::test
{
namespace trait
{
template<typename TDev>
struct EventHostManualTriggerType;

template<typename TDev>
struct IsEventHostManualTriggerSupported;
} 

template<typename TDev>
using EventHostManualTrigger = typename trait::EventHostManualTriggerType<TDev>::type;

template<typename TDev>
ALPAKA_FN_HOST auto isEventHostManualTriggerSupported(TDev const& dev) -> bool
{
return trait::IsEventHostManualTriggerSupported<TDev>::isSupported(dev);
}

namespace cpu::detail
{
template<class TDev = DevCpu>
class EventHostManualTriggerCpuImpl
{
public:
ALPAKA_FN_HOST EventHostManualTriggerCpuImpl(TDev dev) noexcept
: m_dev(std::move(dev))
, m_mutex()
, m_enqueueCount(0u)
, m_bIsReady(true)
{
}
EventHostManualTriggerCpuImpl(EventHostManualTriggerCpuImpl const& other) = delete;
auto operator=(EventHostManualTriggerCpuImpl const&) -> EventHostManualTriggerCpuImpl& = delete;

void trigger()
{
{
std::unique_lock<std::mutex> lock(m_mutex);
m_bIsReady = true;
}
m_conditionVariable.notify_one();
}

public:
TDev const m_dev; 

mutable std::mutex m_mutex; 

mutable std::condition_variable m_conditionVariable; 
std::size_t m_enqueueCount; 

bool m_bIsReady; 
};
} 

template<class TDev = DevCpu>
class EventHostManualTriggerCpu
{
public:
ALPAKA_FN_HOST EventHostManualTriggerCpu(TDev const& dev)
: m_spEventImpl(std::make_shared<cpu::detail::EventHostManualTriggerCpuImpl<TDev>>(dev))
{
}
ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCpu const& rhs) const -> bool
{
return (m_spEventImpl == rhs.m_spEventImpl);
}
ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCpu const& rhs) const -> bool
{
return !((*this) == rhs);
}

void trigger()
{
m_spEventImpl->trigger();
}

public:
std::shared_ptr<cpu::detail::EventHostManualTriggerCpuImpl<TDev>> m_spEventImpl;
};

namespace trait
{
template<>
struct EventHostManualTriggerType<DevCpu>
{
using type = test::EventHostManualTriggerCpu<DevCpu>;
};

template<>
struct IsEventHostManualTriggerSupported<DevCpu>
{
ALPAKA_FN_HOST static auto isSupported(DevCpu const&) -> bool
{
return true;
}
};
} 
} 

namespace alpaka::trait
{
template<typename TDev>
struct GetDev<test::EventHostManualTriggerCpu<TDev>>
{
ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerCpu<TDev> const& event) -> TDev
{
return event.m_spEventImpl->m_dev;
}
};

template<typename TDev>
struct IsComplete<test::EventHostManualTriggerCpu<TDev>>
{
ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerCpu<TDev> const& event) -> bool
{
std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

return event.m_spEventImpl->m_bIsReady;
}
};

template<typename TDev>
struct Enqueue<QueueGenericThreadsNonBlocking<TDev>, test::EventHostManualTriggerCpu<TDev>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueGenericThreadsNonBlocking<TDev>& queue,
test::EventHostManualTriggerCpu<TDev>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl = event.m_spEventImpl;

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

ALPAKA_ASSERT(spEventImpl->m_bIsReady);

spEventImpl->m_bIsReady = false;

++spEventImpl->m_enqueueCount;

auto const enqueueCount = spEventImpl->m_enqueueCount;

queue.m_spQueueImpl->m_workerThread->enqueueTask(
[spEventImpl, enqueueCount]()
{
std::unique_lock<std::mutex> lk2(spEventImpl->m_mutex);
spEventImpl->m_conditionVariable.wait(
lk2,
[spEventImpl, enqueueCount]
{ return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady; });
});
}
};

template<typename TDev>
struct Enqueue<QueueGenericThreadsBlocking<TDev>, test::EventHostManualTriggerCpu<TDev>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueGenericThreadsBlocking<TDev>&,
test::EventHostManualTriggerCpu<TDev>& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl = event.m_spEventImpl;

std::unique_lock<std::mutex> lk(spEventImpl->m_mutex);

ALPAKA_ASSERT(spEventImpl->m_bIsReady);

spEventImpl->m_bIsReady = false;

++spEventImpl->m_enqueueCount;

auto const enqueueCount = spEventImpl->m_enqueueCount;

spEventImpl->m_conditionVariable.wait(
lk,
[spEventImpl, enqueueCount]
{ return (enqueueCount != spEventImpl->m_enqueueCount) || spEventImpl->m_bIsReady; });
}
};
} 

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#    include <alpaka/core/BoostPredef.hpp>

#    include <cuda.h>

#    if !BOOST_LANG_CUDA && !defined(ALPAKA_HOST_ONLY)
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    include <alpaka/core/Cuda.hpp>


namespace alpaka::test
{
namespace uniform_cuda_hip::detail
{
class EventHostManualTriggerCudaImpl final
{
using TApi = alpaka::ApiCudaRt;

public:
ALPAKA_FN_HOST EventHostManualTriggerCudaImpl(DevCudaRt const& dev)
: m_dev(dev)
, m_mutex()
, m_bIsReady(true)
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaMalloc(&m_devMem, static_cast<size_t>(sizeof(int32_t))));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
cudaMemset(m_devMem, static_cast<int>(0u), static_cast<size_t>(sizeof(int32_t))));
}
EventHostManualTriggerCudaImpl(EventHostManualTriggerCudaImpl const&) = delete;
auto operator=(EventHostManualTriggerCudaImpl const&) -> EventHostManualTriggerCudaImpl& = delete;
ALPAKA_FN_HOST ~EventHostManualTriggerCudaImpl()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(cudaFree(m_devMem));
}

void trigger()
{
std::unique_lock<std::mutex> lock(m_mutex);
m_bIsReady = true;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(cudaSetDevice(m_dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
cudaMemset(m_devMem, static_cast<int>(1u), static_cast<size_t>(sizeof(int32_t))));
}

public:
DevCudaRt const m_dev; 

mutable std::mutex m_mutex; 
void* m_devMem;

bool m_bIsReady; 
};
} 

class EventHostManualTriggerCuda final
{
public:
ALPAKA_FN_HOST EventHostManualTriggerCuda(DevCudaRt const& dev)
: m_spEventImpl(std::make_shared<uniform_cuda_hip::detail::EventHostManualTriggerCudaImpl>(dev))
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
}
ALPAKA_FN_HOST auto operator==(EventHostManualTriggerCuda const& rhs) const -> bool
{
return (m_spEventImpl == rhs.m_spEventImpl);
}
ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerCuda const& rhs) const -> bool
{
return !((*this) == rhs);
}

void trigger()
{
m_spEventImpl->trigger();
}

public:
std::shared_ptr<uniform_cuda_hip::detail::EventHostManualTriggerCudaImpl> m_spEventImpl;
};

namespace trait
{
template<>
struct EventHostManualTriggerType<DevCudaRt>
{
using type = test::EventHostManualTriggerCuda;
};
template<>
struct IsEventHostManualTriggerSupported<DevCudaRt>
{
ALPAKA_FN_HOST static auto isSupported(DevCudaRt const& dev) -> bool
{
#    if CUDA_VERSION < 11070
int result = 0;
cuDeviceGetAttribute(&result, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev.getNativeHandle());
return result != 0;
#    else
return true; 
#    endif
}
};
} 
} 

namespace alpaka::trait
{
namespace detail
{
inline auto streamWaitValue(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
-> CUresult
{
#    if(CUDA_VERSION < 11070) || (CUDA_VERSION >= 12000)
return cuStreamWaitValue32(stream, addr, value, flags);
#    else
return cuStreamWaitValue32_v2(stream, addr, value, flags);
#    endif
}
} 

template<>
struct GetDev<test::EventHostManualTriggerCuda>
{
ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerCuda const& event) -> DevCudaRt
{
return event.m_spEventImpl->m_dev;
}
};

template<>
struct IsComplete<test::EventHostManualTriggerCuda>
{
ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerCuda const& event) -> bool
{
std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

return event.m_spEventImpl->m_bIsReady;
}
};

template<>
struct Enqueue<QueueCudaRtNonBlocking, test::EventHostManualTriggerCuda>
{
ALPAKA_FN_HOST static auto enqueue(QueueCudaRtNonBlocking& queue, test::EventHostManualTriggerCuda& event)
-> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl(event.m_spEventImpl);

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

ALPAKA_ASSERT(spEventImpl->m_bIsReady);

spEventImpl->m_bIsReady = false;

ALPAKA_CUDA_DRV_CHECK(detail::streamWaitValue(
static_cast<CUstream>(queue.getNativeHandle()),
reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
0x01010101u,
CU_STREAM_WAIT_VALUE_GEQ));
}
};
template<>
struct Enqueue<QueueCudaRtBlocking, test::EventHostManualTriggerCuda>
{
ALPAKA_FN_HOST static auto enqueue(QueueCudaRtBlocking& queue, test::EventHostManualTriggerCuda& event) -> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl(event.m_spEventImpl);

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

ALPAKA_ASSERT(spEventImpl->m_bIsReady);

spEventImpl->m_bIsReady = false;

ALPAKA_CUDA_DRV_CHECK(detail::streamWaitValue(
static_cast<CUstream>(queue.getNativeHandle()),
reinterpret_cast<CUdeviceptr>(event.m_spEventImpl->m_devMem),
0x01010101u,
CU_STREAM_WAIT_VALUE_GEQ));
}
};
} 
#endif


#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#    include <hip/hip_runtime.h>

#    if !BOOST_LANG_HIP && !defined(ALPAKA_HOST_ONLY)
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/core/Hip.hpp>

namespace alpaka::test
{
namespace hip::detail
{
class EventHostManualTriggerHipImpl final
{
using TApi = alpaka::ApiHipRt;

public:
ALPAKA_FN_HOST EventHostManualTriggerHipImpl(DevHipRt const& dev) : m_dev(dev), m_mutex(), m_bIsReady(true)
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipMalloc(&m_devMem, static_cast<size_t>(sizeof(int32_t))));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
hipMemset(m_devMem, static_cast<int>(0u), static_cast<size_t>(sizeof(int32_t))));
}
EventHostManualTriggerHipImpl(EventHostManualTriggerHipImpl const&) = delete;
auto operator=(EventHostManualTriggerHipImpl const&) -> EventHostManualTriggerHipImpl& = delete;
ALPAKA_FN_HOST ~EventHostManualTriggerHipImpl()
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(hipFree(m_devMem));
}

void trigger()
{
std::unique_lock<std::mutex> lock(m_mutex);
m_bIsReady = true;

ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipSetDevice(m_dev.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
hipMemset(m_devMem, static_cast<int>(1u), static_cast<size_t>(sizeof(int32_t))));
}

public:
DevHipRt const m_dev; 

mutable std::mutex m_mutex; 
void* m_devMem;

bool m_bIsReady; 
};
} 

class EventHostManualTriggerHip final
{
public:
ALPAKA_FN_HOST EventHostManualTriggerHip(DevHipRt const& dev)
: m_spEventImpl(std::make_shared<hip::detail::EventHostManualTriggerHipImpl>(dev))
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
}
ALPAKA_FN_HOST auto operator==(EventHostManualTriggerHip const& rhs) const -> bool
{
return (m_spEventImpl == rhs.m_spEventImpl);
}
ALPAKA_FN_HOST auto operator!=(EventHostManualTriggerHip const& rhs) const -> bool
{
return !((*this) == rhs);
}

void trigger()
{
m_spEventImpl->trigger();
}

public:
std::shared_ptr<hip::detail::EventHostManualTriggerHipImpl> m_spEventImpl;
};

namespace trait
{
template<>
struct EventHostManualTriggerType<DevHipRt>
{
using type = test::EventHostManualTriggerHip;
};

template<>
struct IsEventHostManualTriggerSupported<DevHipRt>
{
ALPAKA_FN_HOST static auto isSupported(DevHipRt const&) -> bool
{
return false;
}
};
} 
} 

namespace alpaka::trait
{
template<>
struct GetDev<test::EventHostManualTriggerHip>
{
ALPAKA_FN_HOST static auto getDev(test::EventHostManualTriggerHip const& event) -> DevHipRt
{
return event.m_spEventImpl->m_dev;
}
};

template<>
struct IsComplete<test::EventHostManualTriggerHip>
{
ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerHip const& event) -> bool
{
std::lock_guard<std::mutex> lk(event.m_spEventImpl->m_mutex);

return event.m_spEventImpl->m_bIsReady;
}
};

template<>
struct Enqueue<QueueHipRtNonBlocking, test::EventHostManualTriggerHip>
{
using TApi = alpaka::ApiHipRt;

ALPAKA_FN_HOST static auto enqueue(QueueHipRtNonBlocking& queue, test::EventHostManualTriggerHip& event)
-> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl(event.m_spEventImpl);

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

ALPAKA_ASSERT(spEventImpl->m_bIsReady);

spEventImpl->m_bIsReady = false;

int32_t hostMem = 0;
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
std::cerr << "[Workaround] polling of device-located value in stream, as hipStreamWaitValue32 is not "
"available.\n";
#    endif
while(hostMem < 0x01010101)
{
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipMemcpyDtoHAsync(
&hostMem,
reinterpret_cast<hipDeviceptr_t>(event.m_spEventImpl->m_devMem),
sizeof(int32_t),
queue.getNativeHandle()));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(hipStreamSynchronize(queue.getNativeHandle()));
}
}
};

template<>
struct Enqueue<QueueHipRtBlocking, test::EventHostManualTriggerHip>
{
using TApi = alpaka::ApiHipRt;

ALPAKA_FN_HOST static auto enqueue(QueueHipRtBlocking& , test::EventHostManualTriggerHip& event)
-> void
{
ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

auto spEventImpl(event.m_spEventImpl);

std::lock_guard<std::mutex> lk(spEventImpl->m_mutex);

ALPAKA_ASSERT(spEventImpl->m_bIsReady);

spEventImpl->m_bIsReady = false;


std::uint32_t hmem = 0;
do
{
std::this_thread::sleep_for(std::chrono::milliseconds(10u));
ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
hipMemcpy(&hmem, event.m_spEventImpl->m_devMem, sizeof(std::uint32_t), hipMemcpyDefault));
} while(hmem < 0x01010101u);
}
};
} 
#endif

#ifdef ALPAKA_ACC_SYCL_ENABLED
namespace alpaka
{
namespace test
{
template<typename TPltf>
class EventHostManualTriggerSycl
{
public:
EventHostManualTriggerSycl(DevGenericSycl<TPltf> const&)
{
}

auto trigger()
{
}
};

namespace trait
{
template<typename TPltf>
struct EventHostManualTriggerType<DevGenericSycl<TPltf>>
{
using type = alpaka::test::EventHostManualTriggerSycl<TPltf>;
};

template<typename TPltf>
struct IsEventHostManualTriggerSupported<DevGenericSycl<TPltf>>
{
ALPAKA_FN_HOST static auto isSupported(DevGenericSycl<TPltf> const&) -> bool
{
return false;
}
};
} 
} 

namespace trait
{
template<typename TPltf>
struct Enqueue<QueueGenericSyclBlocking<DevGenericSycl<TPltf>>, test::EventHostManualTriggerSycl<TPltf>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueGenericSyclBlocking<DevGenericSycl<TPltf>>& queue,
test::EventHostManualTriggerSycl<TPltf>& event) -> void
{
}
};

template<typename TPltf>
struct Enqueue<QueueGenericSyclNonBlocking<DevGenericSycl<TPltf>>, test::EventHostManualTriggerSycl<TPltf>>
{
ALPAKA_FN_HOST static auto enqueue(
QueueGenericSyclNonBlocking<DevGenericSycl<TPltf>>& queue,
test::EventHostManualTriggerSycl<TPltf>& event) -> void
{
}
};

template<typename TPltf>
struct IsComplete<test::EventHostManualTriggerSycl<TPltf>>
{
ALPAKA_FN_HOST static auto isComplete(test::EventHostManualTriggerSycl<TPltf> const& event) -> bool
{
return true;
}
};
} 
} 
#endif
