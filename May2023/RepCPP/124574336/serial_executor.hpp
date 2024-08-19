
#ifndef BOOST_THREAD_SERIAL_EXECUTOR_HPP
#define BOOST_THREAD_SERIAL_EXECUTOR_HPP

#include <boost/thread/detail/config.hpp>
#if defined BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION && defined BOOST_THREAD_PROVIDES_EXECUTORS && defined BOOST_THREAD_USES_MOVE

#include <exception>
#include <boost/thread/detail/delete.hpp>
#include <boost/thread/detail/move.hpp>
#include <boost/thread/concurrent_queues/sync_queue.hpp>
#include <boost/thread/executors/work.hpp>
#include <boost/thread/executors/generic_executor_ref.hpp>
#include <boost/thread/future.hpp>
#include <boost/thread/scoped_thread.hpp>

#include <boost/config/abi_prefix.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4355) 
#endif

namespace boost
{
namespace executors
{
class serial_executor
{
public:
typedef  executors::work work;
private:
typedef  scoped_thread<> thread_t;

concurrent::sync_queue<work > work_queue;
generic_executor_ref ex;
thread_t thr;

struct try_executing_one_task {
work& task;
boost::promise<void> &p;
try_executing_one_task(work& task, boost::promise<void> &p)
: task(task), p(p) {}
void operator()() {
try {
task();
p.set_value();
} catch (...)
{
p.set_exception(current_exception());
}
}
};
public:

generic_executor_ref& underlying_executor() BOOST_NOEXCEPT { return ex; }


bool try_executing_one()
{
work task;
try
{
if (work_queue.try_pull(task) == queue_op_status::success)
{
boost::promise<void> p;
try_executing_one_task tmp(task,p);
ex.submit(tmp);
p.get_future().wait();
return true;
}
return false;
}
catch (...)
{
std::terminate();
}
}
private:

void schedule_one_or_yield()
{
if ( ! try_executing_one())
{
this_thread::yield();
}
}


void worker_thread()
{
while (!closed())
{
schedule_one_or_yield();
}
while (try_executing_one())
{
}
}

public:
BOOST_THREAD_NO_COPYABLE(serial_executor)


template <class Executor>
serial_executor(Executor& ex)
: ex(ex), thr(&serial_executor::worker_thread, this)
{
}

~serial_executor()
{
close();
}


void close()
{
work_queue.close();
}


bool closed()
{
return work_queue.closed();
}


void submit(BOOST_THREAD_RV_REF(work) closure)
{
work_queue.push(boost::move(closure));
}

#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
template <typename Closure>
void submit(Closure & closure)
{
submit(work(closure));
}
#endif
void submit(void (*closure)())
{
submit(work(closure));
}

template <typename Closure>
void submit(BOOST_THREAD_FWD_REF(Closure) closure)
{
work w((boost::forward<Closure>(closure)));
submit(boost::move(w));
}


template <typename Pred>
bool reschedule_until(Pred const& pred)
{
do {
if ( ! try_executing_one())
{
return false;
}
} while (! pred());
return true;
}

};
}
using executors::serial_executor;
}

#if defined(BOOST_MSVC)
# pragma warning(pop)
#endif

#include <boost/config/abi_suffix.hpp>

#endif
#endif
