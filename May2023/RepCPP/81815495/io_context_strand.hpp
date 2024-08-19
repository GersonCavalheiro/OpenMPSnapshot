
#ifndef ASIO_IO_CONTEXT_STRAND_HPP
#define ASIO_IO_CONTEXT_STRAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_EXTENSIONS) \
&& !defined(ASIO_NO_TS_EXECUTORS)

#include "asio/async_result.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/strand_service.hpp"
#include "asio/detail/wrapped_handler.hpp"
#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {


class io_context::strand
{
public:

explicit strand(asio::io_context& io_context)
: service_(asio::use_service<
asio::detail::strand_service>(io_context))
{
service_.construct(impl_);
}


~strand()
{
}

asio::io_context& context() const ASIO_NOEXCEPT
{
return service_.get_io_context();
}


void on_work_started() const ASIO_NOEXCEPT
{
context().get_executor().on_work_started();
}


void on_work_finished() const ASIO_NOEXCEPT
{
context().get_executor().on_work_finished();
}


template <typename Function, typename Allocator>
void dispatch(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
typename decay<Function>::type tmp(ASIO_MOVE_CAST(Function)(f));
service_.dispatch(impl_, tmp);
(void)a;
}

#if !defined(ASIO_NO_DEPRECATED)

template <typename LegacyCompletionHandler>
ASIO_INITFN_AUTO_RESULT_TYPE(LegacyCompletionHandler, void ())
dispatch(ASIO_MOVE_ARG(LegacyCompletionHandler) handler)
{
return async_initiate<LegacyCompletionHandler, void ()>(
initiate_dispatch(), handler, this);
}
#endif 


template <typename Function, typename Allocator>
void post(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
typename decay<Function>::type tmp(ASIO_MOVE_CAST(Function)(f));
service_.post(impl_, tmp);
(void)a;
}

#if !defined(ASIO_NO_DEPRECATED)

template <typename LegacyCompletionHandler>
ASIO_INITFN_AUTO_RESULT_TYPE(LegacyCompletionHandler, void ())
post(ASIO_MOVE_ARG(LegacyCompletionHandler) handler)
{
return async_initiate<LegacyCompletionHandler, void ()>(
initiate_post(), handler, this);
}
#endif 


template <typename Function, typename Allocator>
void defer(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
typename decay<Function>::type tmp(ASIO_MOVE_CAST(Function)(f));
service_.post(impl_, tmp);
(void)a;
}

#if !defined(ASIO_NO_DEPRECATED)

template <typename Handler>
#if defined(GENERATING_DOCUMENTATION)
unspecified
#else
detail::wrapped_handler<strand, Handler, detail::is_continuation_if_running>
#endif
wrap(Handler handler)
{
return detail::wrapped_handler<io_context::strand, Handler,
detail::is_continuation_if_running>(*this, handler);
}
#endif 


bool running_in_this_thread() const ASIO_NOEXCEPT
{
return service_.running_in_this_thread(impl_);
}


friend bool operator==(const strand& a, const strand& b) ASIO_NOEXCEPT
{
return a.impl_ == b.impl_;
}


friend bool operator!=(const strand& a, const strand& b) ASIO_NOEXCEPT
{
return a.impl_ != b.impl_;
}

private:
#if !defined(ASIO_NO_DEPRECATED)
struct initiate_dispatch
{
template <typename LegacyCompletionHandler>
void operator()(ASIO_MOVE_ARG(LegacyCompletionHandler) handler,
strand* self) const
{
ASIO_LEGACY_COMPLETION_HANDLER_CHECK(
LegacyCompletionHandler, handler) type_check;

detail::non_const_lvalue<LegacyCompletionHandler> handler2(handler);
self->service_.dispatch(self->impl_, handler2.value);
}
};

struct initiate_post
{
template <typename LegacyCompletionHandler>
void operator()(ASIO_MOVE_ARG(LegacyCompletionHandler) handler,
strand* self) const
{
ASIO_LEGACY_COMPLETION_HANDLER_CHECK(
LegacyCompletionHandler, handler) type_check;

detail::non_const_lvalue<LegacyCompletionHandler> handler2(handler);
self->service_.post(self->impl_, handler2.value);
}
};
#endif 

asio::detail::strand_service& service_;
mutable asio::detail::strand_service::implementation_type impl_;
};

} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
