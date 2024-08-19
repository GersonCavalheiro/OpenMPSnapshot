
#ifndef BOOST_ASIO_IO_CONTEXT_STRAND_HPP
#define BOOST_ASIO_IO_CONTEXT_STRAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_NO_EXTENSIONS) \
&& !defined(BOOST_ASIO_NO_TS_EXECUTORS)

#include <boost/asio/async_result.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/strand_service.hpp>
#include <boost/asio/detail/wrapped_handler.hpp>
#include <boost/asio/io_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {


class io_context::strand
{
public:

explicit strand(boost::asio::io_context& io_context)
: service_(boost::asio::use_service<
boost::asio::detail::strand_service>(io_context))
{
service_.construct(impl_);
}


~strand()
{
}

boost::asio::io_context& context() const BOOST_ASIO_NOEXCEPT
{
return service_.get_io_context();
}


void on_work_started() const BOOST_ASIO_NOEXCEPT
{
context().get_executor().on_work_started();
}


void on_work_finished() const BOOST_ASIO_NOEXCEPT
{
context().get_executor().on_work_finished();
}


template <typename Function, typename Allocator>
void dispatch(BOOST_ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
typename decay<Function>::type tmp(BOOST_ASIO_MOVE_CAST(Function)(f));
service_.dispatch(impl_, tmp);
(void)a;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

template <typename LegacyCompletionHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(LegacyCompletionHandler, void ())
dispatch(BOOST_ASIO_MOVE_ARG(LegacyCompletionHandler) handler)
{
return async_initiate<LegacyCompletionHandler, void ()>(
initiate_dispatch(), handler, this);
}
#endif 


template <typename Function, typename Allocator>
void post(BOOST_ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
typename decay<Function>::type tmp(BOOST_ASIO_MOVE_CAST(Function)(f));
service_.post(impl_, tmp);
(void)a;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

template <typename LegacyCompletionHandler>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(LegacyCompletionHandler, void ())
post(BOOST_ASIO_MOVE_ARG(LegacyCompletionHandler) handler)
{
return async_initiate<LegacyCompletionHandler, void ()>(
initiate_post(), handler, this);
}
#endif 


template <typename Function, typename Allocator>
void defer(BOOST_ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
typename decay<Function>::type tmp(BOOST_ASIO_MOVE_CAST(Function)(f));
service_.post(impl_, tmp);
(void)a;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

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


bool running_in_this_thread() const BOOST_ASIO_NOEXCEPT
{
return service_.running_in_this_thread(impl_);
}


friend bool operator==(const strand& a, const strand& b) BOOST_ASIO_NOEXCEPT
{
return a.impl_ == b.impl_;
}


friend bool operator!=(const strand& a, const strand& b) BOOST_ASIO_NOEXCEPT
{
return a.impl_ != b.impl_;
}

private:
#if !defined(BOOST_ASIO_NO_DEPRECATED)
struct initiate_dispatch
{
template <typename LegacyCompletionHandler>
void operator()(BOOST_ASIO_MOVE_ARG(LegacyCompletionHandler) handler,
strand* self) const
{
BOOST_ASIO_LEGACY_COMPLETION_HANDLER_CHECK(
LegacyCompletionHandler, handler) type_check;

detail::non_const_lvalue<LegacyCompletionHandler> handler2(handler);
self->service_.dispatch(self->impl_, handler2.value);
}
};

struct initiate_post
{
template <typename LegacyCompletionHandler>
void operator()(BOOST_ASIO_MOVE_ARG(LegacyCompletionHandler) handler,
strand* self) const
{
BOOST_ASIO_LEGACY_COMPLETION_HANDLER_CHECK(
LegacyCompletionHandler, handler) type_check;

detail::non_const_lvalue<LegacyCompletionHandler> handler2(handler);
self->service_.post(self->impl_, handler2.value);
}
};
#endif 

boost::asio::detail::strand_service& service_;
mutable boost::asio::detail::strand_service::implementation_type impl_;
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
