

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/event/Traits.hpp>
#    include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#    include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>
#    include <alpaka/wait/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <functional>
#    include <memory>
#    include <stdexcept>

namespace alpaka
{
template<typename TDev>
class EventGenericSycl final
{
public:
explicit EventGenericSycl(TDev const& dev) : m_dev{dev}
{
}

friend auto operator==(EventGenericSycl const& lhs, EventGenericSycl const& rhs) -> bool
{
return (lhs.m_event == rhs.m_event);
}

friend auto operator!=(EventGenericSycl const& lhs, EventGenericSycl const& rhs) -> bool
{
return !(lhs == rhs);
}

[[nodiscard]] auto getNativeHandle() const
{
return m_event;
}

void setEvent(sycl::event const& event)
{
m_event = event;
}

TDev m_dev;

private:
sycl::event m_event{};
};
} 

namespace alpaka::trait
{
template<typename TDev>
struct GetDev<EventGenericSycl<TDev>>
{
static auto getDev(EventGenericSycl<TDev> const& event) -> TDev
{
return event.m_dev;
}
};

template<typename TDev>
struct IsComplete<EventGenericSycl<TDev>>
{
static auto isComplete(EventGenericSycl<TDev> const& event)
{
auto const status
= event.getNativeHandle().template get_info<sycl::info::event::command_execution_status>();
return (status == sycl::info::event_command_status::complete);
}
};

template<typename TDev>
struct Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
{
static auto enqueue(QueueGenericSyclNonBlocking<TDev>& queue, EventGenericSycl<TDev>& event)
{
event.setEvent(queue.m_impl->get_last_event());
}
};

template<typename TDev>
struct Enqueue<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>
{
static auto enqueue(QueueGenericSyclBlocking<TDev>& queue, EventGenericSycl<TDev>& event)
{
event.setEvent(queue.m_impl->get_last_event());
}
};

template<typename TDev>
struct CurrentThreadWaitFor<EventGenericSycl<TDev>>
{
static auto currentThreadWaitFor(EventGenericSycl<TDev> const& event)
{
event.getNativeHandle().wait_and_throw();
}
};

template<typename TDev>
struct WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
{
static auto waiterWaitFor(QueueGenericSyclNonBlocking<TDev>& queue, EventGenericSycl<TDev> const& event)
{
queue.m_impl->register_dependency(event.getNativeHandle());
}
};

template<typename TDev>
struct WaiterWaitFor<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>
{
static auto waiterWaitFor(QueueGenericSyclBlocking<TDev>& queue, EventGenericSycl<TDev> const& event)
{
queue.m_impl->register_dependency(event.getNativeHandle());
}
};

template<typename TDev>
struct WaiterWaitFor<TDev, EventGenericSycl<TDev>>
{
static auto waiterWaitFor(TDev& dev, EventGenericSycl<TDev> const& event)
{
dev.m_impl->register_dependency(event.getNativeHandle());
}
};

template<typename TDev>
struct NativeHandle<EventGenericSycl<TDev>>
{
[[nodiscard]] static auto getNativeHandle(EventGenericSycl<TDev> const& event)
{
return event.getNativeHandle();
}
};
} 

#endif
