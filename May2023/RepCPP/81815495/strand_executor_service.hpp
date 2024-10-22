
#ifndef ASIO_DETAIL_IMPL_STRAND_EXECUTOR_SERVICE_HPP
#define ASIO_DETAIL_IMPL_STRAND_EXECUTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/recycling_allocator.hpp"
#include "asio/executor_work_guard.hpp"
#include "asio/defer.hpp"
#include "asio/dispatch.hpp"
#include "asio/post.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename F, typename Allocator>
class strand_executor_service::allocator_binder
{
public:
typedef Allocator allocator_type;

allocator_binder(ASIO_MOVE_ARG(F) f, const Allocator& a)
: f_(ASIO_MOVE_CAST(F)(f)),
allocator_(a)
{
}

allocator_binder(const allocator_binder& other)
: f_(other.f_),
allocator_(other.allocator_)
{
}

#if defined(ASIO_HAS_MOVE)
allocator_binder(allocator_binder&& other)
: f_(ASIO_MOVE_CAST(F)(other.f_)),
allocator_(ASIO_MOVE_CAST(allocator_type)(other.allocator_))
{
}
#endif 

allocator_type get_allocator() const ASIO_NOEXCEPT
{
return allocator_;
}

void operator()()
{
f_();
}

private:
F f_;
allocator_type allocator_;
};

template <typename Executor>
class strand_executor_service::invoker<Executor,
typename enable_if<
execution::is_executor<Executor>::value
>::type>
{
public:
invoker(const implementation_type& impl, Executor& ex)
: impl_(impl),
executor_(asio::prefer(ex, execution::outstanding_work.tracked))
{
}

invoker(const invoker& other)
: impl_(other.impl_),
executor_(other.executor_)
{
}

#if defined(ASIO_HAS_MOVE)
invoker(invoker&& other)
: impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_)),
executor_(ASIO_MOVE_CAST(executor_type)(other.executor_))
{
}
#endif 

struct on_invoker_exit
{
invoker* this_;

~on_invoker_exit()
{
if (push_waiting_to_ready(this_->impl_))
{
recycling_allocator<void> allocator;
executor_type ex = this_->executor_;
execution::execute(
asio::prefer(
asio::require(
ASIO_MOVE_CAST(executor_type)(ex),
execution::blocking.never),
execution::allocator(allocator)),
ASIO_MOVE_CAST(invoker)(*this_));
}
}
};

void operator()()
{
on_invoker_exit on_exit = { this };
(void)on_exit;

run_ready_handlers(impl_);
}

private:
typedef typename decay<
typename prefer_result<
Executor,
execution::outstanding_work_t::tracked_t
>::type
>::type executor_type;

implementation_type impl_;
executor_type executor_;
};

#if !defined(ASIO_NO_TS_EXECUTORS)

template <typename Executor>
class strand_executor_service::invoker<Executor,
typename enable_if<
!execution::is_executor<Executor>::value
>::type>
{
public:
invoker(const implementation_type& impl, Executor& ex)
: impl_(impl),
work_(ex)
{
}

invoker(const invoker& other)
: impl_(other.impl_),
work_(other.work_)
{
}

#if defined(ASIO_HAS_MOVE)
invoker(invoker&& other)
: impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_)),
work_(ASIO_MOVE_CAST(executor_work_guard<Executor>)(other.work_))
{
}
#endif 

struct on_invoker_exit
{
invoker* this_;

~on_invoker_exit()
{
if (push_waiting_to_ready(this_->impl_))
{
Executor ex(this_->work_.get_executor());
recycling_allocator<void> allocator;
ex.post(ASIO_MOVE_CAST(invoker)(*this_), allocator);
}
}
};

void operator()()
{
on_invoker_exit on_exit = { this };
(void)on_exit;

run_ready_handlers(impl_);
}

private:
implementation_type impl_;
executor_work_guard<Executor> work_;
};

#endif 

template <typename Executor, typename Function>
inline void strand_executor_service::execute(const implementation_type& impl,
Executor& ex, ASIO_MOVE_ARG(Function) function,
typename enable_if<
can_query<Executor, execution::allocator_t<void> >::value
>::type*)
{
return strand_executor_service::do_execute(impl, ex,
ASIO_MOVE_CAST(Function)(function),
asio::query(ex, execution::allocator));
}

template <typename Executor, typename Function>
inline void strand_executor_service::execute(const implementation_type& impl,
Executor& ex, ASIO_MOVE_ARG(Function) function,
typename enable_if<
!can_query<Executor, execution::allocator_t<void> >::value
>::type*)
{
return strand_executor_service::do_execute(impl, ex,
ASIO_MOVE_CAST(Function)(function),
std::allocator<void>());
}

template <typename Executor, typename Function, typename Allocator>
void strand_executor_service::do_execute(const implementation_type& impl,
Executor& ex, ASIO_MOVE_ARG(Function) function, const Allocator& a)
{
typedef typename decay<Function>::type function_type;

if (asio::query(ex, execution::blocking) != execution::blocking.never
&& running_in_this_thread(impl))
{
function_type tmp(ASIO_MOVE_CAST(Function)(function));

fenced_block b(fenced_block::full);
asio_handler_invoke_helpers::invoke(tmp, tmp);
return;
}

typedef executor_op<function_type, Allocator> op;
typename op::ptr p = { detail::addressof(a), op::ptr::allocate(a), 0 };
p.p = new (p.v) op(ASIO_MOVE_CAST(Function)(function), a);

ASIO_HANDLER_CREATION((impl->service_->context(), *p.p,
"strand_executor", impl.get(), 0, "execute"));

bool first = enqueue(impl, p.p);
p.v = p.p = 0;
if (first)
{
execution::execute(ex, invoker<Executor>(impl, ex));
}
}

template <typename Executor, typename Function, typename Allocator>
void strand_executor_service::dispatch(const implementation_type& impl,
Executor& ex, ASIO_MOVE_ARG(Function) function, const Allocator& a)
{
typedef typename decay<Function>::type function_type;

if (running_in_this_thread(impl))
{
function_type tmp(ASIO_MOVE_CAST(Function)(function));

fenced_block b(fenced_block::full);
asio_handler_invoke_helpers::invoke(tmp, tmp);
return;
}

typedef executor_op<function_type, Allocator> op;
typename op::ptr p = { detail::addressof(a), op::ptr::allocate(a), 0 };
p.p = new (p.v) op(ASIO_MOVE_CAST(Function)(function), a);

ASIO_HANDLER_CREATION((impl->service_->context(), *p.p,
"strand_executor", impl.get(), 0, "dispatch"));

bool first = enqueue(impl, p.p);
p.v = p.p = 0;
if (first)
{
asio::dispatch(ex,
allocator_binder<invoker<Executor>, Allocator>(
invoker<Executor>(impl, ex), a));
}
}

template <typename Executor, typename Function, typename Allocator>
void strand_executor_service::post(const implementation_type& impl,
Executor& ex, ASIO_MOVE_ARG(Function) function, const Allocator& a)
{
typedef typename decay<Function>::type function_type;

typedef executor_op<function_type, Allocator> op;
typename op::ptr p = { detail::addressof(a), op::ptr::allocate(a), 0 };
p.p = new (p.v) op(ASIO_MOVE_CAST(Function)(function), a);

ASIO_HANDLER_CREATION((impl->service_->context(), *p.p,
"strand_executor", impl.get(), 0, "post"));

bool first = enqueue(impl, p.p);
p.v = p.p = 0;
if (first)
{
asio::post(ex,
allocator_binder<invoker<Executor>, Allocator>(
invoker<Executor>(impl, ex), a));
}
}

template <typename Executor, typename Function, typename Allocator>
void strand_executor_service::defer(const implementation_type& impl,
Executor& ex, ASIO_MOVE_ARG(Function) function, const Allocator& a)
{
typedef typename decay<Function>::type function_type;

typedef executor_op<function_type, Allocator> op;
typename op::ptr p = { detail::addressof(a), op::ptr::allocate(a), 0 };
p.p = new (p.v) op(ASIO_MOVE_CAST(Function)(function), a);

ASIO_HANDLER_CREATION((impl->service_->context(), *p.p,
"strand_executor", impl.get(), 0, "defer"));

bool first = enqueue(impl, p.p);
p.v = p.p = 0;
if (first)
{
asio::defer(ex,
allocator_binder<invoker<Executor>, Allocator>(
invoker<Executor>(impl, ex), a));
}
}

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
