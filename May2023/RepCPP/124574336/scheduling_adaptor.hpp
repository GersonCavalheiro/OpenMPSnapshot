
#ifndef BOOST_THREAD_EXECUTORS_SCHEDULING_ADAPTOR_HPP
#define BOOST_THREAD_EXECUTORS_SCHEDULING_ADAPTOR_HPP

#include <boost/thread/detail/config.hpp>
#if defined BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION && defined BOOST_THREAD_PROVIDES_EXECUTORS && defined BOOST_THREAD_USES_MOVE
#include <boost/thread/executors/detail/scheduled_executor_base.hpp>

#if defined(BOOST_MSVC)
# pragma warning(push)
# pragma warning(disable: 4355) 
#endif

namespace boost
{
namespace executors
{

template <typename Executor>
class scheduling_adaptor : public detail::scheduled_executor_base<>
{
private:
Executor& _exec;
thread _scheduler;
public:

scheduling_adaptor(Executor& ex)
: super(),
_exec(ex),
_scheduler(&super::loop, this) {}

~scheduling_adaptor()
{
this->close();
_scheduler.interrupt();
_scheduler.join();
}

Executor& underlying_executor()
{
return _exec;
}

private:
typedef detail::scheduled_executor_base<> super;
}; 

} 

using executors::scheduling_adaptor;

} 

#if defined(BOOST_MSVC)
# pragma warning(pop)
#endif

#endif
#endif
