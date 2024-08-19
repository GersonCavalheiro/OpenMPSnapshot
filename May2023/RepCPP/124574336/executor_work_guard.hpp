
#ifndef BOOST_ASIO_EXECUTOR_WORK_GUARD_HPP
#define BOOST_ASIO_EXECUTOR_WORK_GUARD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_NO_TS_EXECUTORS)

#include <boost/asio/associated_executor.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/execution.hpp>
#include <boost/asio/is_executor.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if !defined(BOOST_ASIO_EXECUTOR_WORK_GUARD_DECL)
#define BOOST_ASIO_EXECUTOR_WORK_GUARD_DECL

template <typename Executor, typename = void>
class executor_work_guard;

#endif 

#if defined(GENERATING_DOCUMENTATION)
template <typename Executor>
#else 
template <typename Executor, typename>
#endif 
class executor_work_guard
{
public:
typedef Executor executor_type;


explicit executor_work_guard(const executor_type& e) BOOST_ASIO_NOEXCEPT
: executor_(e),
owns_(true)
{
executor_.on_work_started();
}

executor_work_guard(const executor_work_guard& other) BOOST_ASIO_NOEXCEPT
: executor_(other.executor_),
owns_(other.owns_)
{
if (owns_)
executor_.on_work_started();
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
executor_work_guard(executor_work_guard&& other) BOOST_ASIO_NOEXCEPT
: executor_(BOOST_ASIO_MOVE_CAST(Executor)(other.executor_)),
owns_(other.owns_)
{
other.owns_ = false;
}
#endif 


~executor_work_guard()
{
if (owns_)
executor_.on_work_finished();
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return executor_;
}

bool owns_work() const BOOST_ASIO_NOEXCEPT
{
return owns_;
}


void reset() BOOST_ASIO_NOEXCEPT
{
if (owns_)
{
executor_.on_work_finished();
owns_ = false;
}
}

private:
executor_work_guard& operator=(const executor_work_guard&);

executor_type executor_;
bool owns_;
};

#if !defined(GENERATING_DOCUMENTATION)

template <typename Executor>
class executor_work_guard<Executor,
typename enable_if<
!is_executor<Executor>::value && execution::is_executor<Executor>::value
>::type>
{
public:
typedef Executor executor_type;

explicit executor_work_guard(const executor_type& e) BOOST_ASIO_NOEXCEPT
: executor_(e),
owns_(true)
{
new (&work_) work_type(boost::asio::prefer(executor_,
execution::outstanding_work.tracked));
}

executor_work_guard(const executor_work_guard& other) BOOST_ASIO_NOEXCEPT
: executor_(other.executor_),
owns_(other.owns_)
{
if (owns_)
{
new (&work_) work_type(boost::asio::prefer(executor_,
execution::outstanding_work.tracked));
}
}

#if defined(BOOST_ASIO_HAS_MOVE)
executor_work_guard(executor_work_guard&& other) BOOST_ASIO_NOEXCEPT
: executor_(BOOST_ASIO_MOVE_CAST(Executor)(other.executor_)),
owns_(other.owns_)
{
if (owns_)
{
new (&work_) work_type(
BOOST_ASIO_MOVE_CAST(work_type)(
*static_cast<work_type*>(
static_cast<void*>(&other.work_))));
other.owns_ = false;
}
}
#endif 

~executor_work_guard()
{
if (owns_)
static_cast<work_type*>(static_cast<void*>(&work_))->~work_type();
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return executor_;
}

bool owns_work() const BOOST_ASIO_NOEXCEPT
{
return owns_;
}

void reset() BOOST_ASIO_NOEXCEPT
{
if (owns_)
{
static_cast<work_type*>(static_cast<void*>(&work_))->~work_type();
owns_ = false;
}
}

private:
executor_work_guard& operator=(const executor_work_guard&);

typedef typename decay<
typename prefer_result<
const executor_type&,
execution::outstanding_work_t::tracked_t
>::type
>::type work_type;

executor_type executor_;
typename aligned_storage<sizeof(work_type),
alignment_of<work_type>::value>::type work_;
bool owns_;
};

#endif 

template <typename Executor>
inline executor_work_guard<Executor> make_work_guard(const Executor& ex,
typename enable_if<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type* = 0)
{
return executor_work_guard<Executor>(ex);
}

template <typename ExecutionContext>
inline executor_work_guard<typename ExecutionContext::executor_type>
make_work_guard(ExecutionContext& ctx,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
return executor_work_guard<typename ExecutionContext::executor_type>(
ctx.get_executor());
}

template <typename T>
inline executor_work_guard<typename associated_executor<T>::type>
make_work_guard(const T& t,
typename enable_if<
!is_executor<T>::value && !execution::is_executor<T>::value
&& !is_convertible<T&, execution_context&
>::value>::type* = 0)
{
return executor_work_guard<typename associated_executor<T>::type>(
associated_executor<T>::get(t));
}

template <typename T, typename Executor>
inline executor_work_guard<typename associated_executor<T, Executor>::type>
make_work_guard(const T& t, const Executor& ex,
typename enable_if<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type* = 0)
{
return executor_work_guard<typename associated_executor<T, Executor>::type>(
associated_executor<T, Executor>::get(t, ex));
}

template <typename T, typename ExecutionContext>
inline executor_work_guard<typename associated_executor<T,
typename ExecutionContext::executor_type>::type>
make_work_guard(const T& t, ExecutionContext& ctx,
typename enable_if<
!is_executor<T>::value && !execution::is_executor<T>::value
&& !is_convertible<T&, execution_context&>::value
&& is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
return executor_work_guard<typename associated_executor<T,
typename ExecutionContext::executor_type>::type>(
associated_executor<T, typename ExecutionContext::executor_type>::get(
t, ctx.get_executor()));
}

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
