
#ifndef BOOST_ASIO_DETAIL_DEADLINE_TIMER_SERVICE_HPP
#define BOOST_ASIO_DETAIL_DEADLINE_TIMER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/detail/timer_queue.hpp>
#include <boost/asio/detail/timer_queue_ptime.hpp>
#include <boost/asio/detail/timer_scheduler.hpp>
#include <boost/asio/detail/wait_handler.hpp>
#include <boost/asio/detail/wait_op.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
# include <chrono>
# include <thread>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Time_Traits>
class deadline_timer_service
: public execution_context_service_base<deadline_timer_service<Time_Traits> >
{
public:
typedef typename Time_Traits::time_type time_type;

typedef typename Time_Traits::duration_type duration_type;

struct implementation_type
: private boost::asio::detail::noncopyable
{
time_type expiry;
bool might_have_pending_waits;
typename timer_queue<Time_Traits>::per_timer_data timer_data;
};

deadline_timer_service(execution_context& context)
: execution_context_service_base<
deadline_timer_service<Time_Traits> >(context),
scheduler_(boost::asio::use_service<timer_scheduler>(context))
{
scheduler_.init_task();
scheduler_.add_timer_queue(timer_queue_);
}

~deadline_timer_service()
{
scheduler_.remove_timer_queue(timer_queue_);
}

void shutdown()
{
}

void construct(implementation_type& impl)
{
impl.expiry = time_type();
impl.might_have_pending_waits = false;
}

void destroy(implementation_type& impl)
{
boost::system::error_code ec;
cancel(impl, ec);
}

void move_construct(implementation_type& impl,
implementation_type& other_impl)
{
scheduler_.move_timer(timer_queue_, impl.timer_data, other_impl.timer_data);

impl.expiry = other_impl.expiry;
other_impl.expiry = time_type();

impl.might_have_pending_waits = other_impl.might_have_pending_waits;
other_impl.might_have_pending_waits = false;
}

void move_assign(implementation_type& impl,
deadline_timer_service& other_service,
implementation_type& other_impl)
{
if (this != &other_service)
if (impl.might_have_pending_waits)
scheduler_.cancel_timer(timer_queue_, impl.timer_data);

other_service.scheduler_.move_timer(other_service.timer_queue_,
impl.timer_data, other_impl.timer_data);

impl.expiry = other_impl.expiry;
other_impl.expiry = time_type();

impl.might_have_pending_waits = other_impl.might_have_pending_waits;
other_impl.might_have_pending_waits = false;
}

void converting_move_construct(implementation_type& impl,
deadline_timer_service&, implementation_type& other_impl)
{
move_construct(impl, other_impl);
}

void converting_move_assign(implementation_type& impl,
deadline_timer_service& other_service,
implementation_type& other_impl)
{
move_assign(impl, other_service, other_impl);
}

std::size_t cancel(implementation_type& impl, boost::system::error_code& ec)
{
if (!impl.might_have_pending_waits)
{
ec = boost::system::error_code();
return 0;
}

BOOST_ASIO_HANDLER_OPERATION((scheduler_.context(),
"deadline_timer", &impl, 0, "cancel"));

std::size_t count = scheduler_.cancel_timer(timer_queue_, impl.timer_data);
impl.might_have_pending_waits = false;
ec = boost::system::error_code();
return count;
}

std::size_t cancel_one(implementation_type& impl,
boost::system::error_code& ec)
{
if (!impl.might_have_pending_waits)
{
ec = boost::system::error_code();
return 0;
}

BOOST_ASIO_HANDLER_OPERATION((scheduler_.context(),
"deadline_timer", &impl, 0, "cancel_one"));

std::size_t count = scheduler_.cancel_timer(
timer_queue_, impl.timer_data, 1);
if (count == 0)
impl.might_have_pending_waits = false;
ec = boost::system::error_code();
return count;
}

time_type expiry(const implementation_type& impl) const
{
return impl.expiry;
}

time_type expires_at(const implementation_type& impl) const
{
return impl.expiry;
}

duration_type expires_from_now(const implementation_type& impl) const
{
return Time_Traits::subtract(this->expiry(impl), Time_Traits::now());
}

std::size_t expires_at(implementation_type& impl,
const time_type& expiry_time, boost::system::error_code& ec)
{
std::size_t count = cancel(impl, ec);
impl.expiry = expiry_time;
ec = boost::system::error_code();
return count;
}

std::size_t expires_after(implementation_type& impl,
const duration_type& expiry_time, boost::system::error_code& ec)
{
return expires_at(impl,
Time_Traits::add(Time_Traits::now(), expiry_time), ec);
}

std::size_t expires_from_now(implementation_type& impl,
const duration_type& expiry_time, boost::system::error_code& ec)
{
return expires_at(impl,
Time_Traits::add(Time_Traits::now(), expiry_time), ec);
}

void wait(implementation_type& impl, boost::system::error_code& ec)
{
time_type now = Time_Traits::now();
ec = boost::system::error_code();
while (Time_Traits::less_than(now, impl.expiry) && !ec)
{
this->do_wait(Time_Traits::to_posix_duration(
Time_Traits::subtract(impl.expiry, now)), ec);
now = Time_Traits::now();
}
}

template <typename Handler, typename IoExecutor>
void async_wait(implementation_type& impl,
Handler& handler, const IoExecutor& io_ex)
{
typedef wait_handler<Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(handler, io_ex);

impl.might_have_pending_waits = true;

BOOST_ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "deadline_timer", &impl, 0, "async_wait"));

scheduler_.schedule_timer(timer_queue_, impl.expiry, impl.timer_data, p.p);
p.v = p.p = 0;
}

private:
template <typename Duration>
void do_wait(const Duration& timeout, boost::system::error_code& ec)
{
#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
std::this_thread::sleep_for(
std::chrono::seconds(timeout.total_seconds())
+ std::chrono::microseconds(timeout.total_microseconds()));
ec = boost::system::error_code();
#else 
::timeval tv;
tv.tv_sec = timeout.total_seconds();
tv.tv_usec = timeout.total_microseconds() % 1000000;
socket_ops::select(0, 0, 0, 0, &tv, ec);
#endif 
}

timer_queue<Time_Traits> timer_queue_;

timer_scheduler& scheduler_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
