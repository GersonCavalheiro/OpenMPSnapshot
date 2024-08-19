
#ifndef BOOST_ASIO_POSIX_BASIC_DESCRIPTOR_HPP
#define BOOST_ASIO_POSIX_BASIC_DESCRIPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/async_result.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/io_object_impl.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/reactive_descriptor_service.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/posix/descriptor_base.hpp>

#if defined(BOOST_ASIO_HAS_MOVE)
# include <utility>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace posix {


template <typename Executor = any_io_executor>
class basic_descriptor
: public descriptor_base
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_descriptor<Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#else
typedef detail::reactive_descriptor_service::native_handle_type
native_handle_type;
#endif

typedef basic_descriptor lowest_layer_type;


explicit basic_descriptor(const executor_type& ex)
: impl_(ex)
{
}


template <typename ExecutionContext>
explicit basic_descriptor(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
}


basic_descriptor(const executor_type& ex,
const native_handle_type& native_descriptor)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_descriptor, ec);
boost::asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_descriptor(ExecutionContext& context,
const native_handle_type& native_descriptor,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_descriptor, ec);
boost::asio::detail::throw_error(ec, "assign");
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_descriptor(basic_descriptor&& other) BOOST_ASIO_NOEXCEPT
: impl_(std::move(other.impl_))
{
}


basic_descriptor& operator=(basic_descriptor&& other)
{
impl_ = std::move(other.impl_);
return *this;
}
#endif 

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return impl_.get_executor();
}


lowest_layer_type& lowest_layer()
{
return *this;
}


const lowest_layer_type& lowest_layer() const
{
return *this;
}


void assign(const native_handle_type& native_descriptor)
{
boost::system::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_descriptor, ec);
boost::asio::detail::throw_error(ec, "assign");
}


BOOST_ASIO_SYNC_OP_VOID assign(const native_handle_type& native_descriptor,
boost::system::error_code& ec)
{
impl_.get_service().assign(
impl_.get_implementation(), native_descriptor, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}

bool is_open() const
{
return impl_.get_service().is_open(impl_.get_implementation());
}


void close()
{
boost::system::error_code ec;
impl_.get_service().close(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "close");
}


BOOST_ASIO_SYNC_OP_VOID close(boost::system::error_code& ec)
{
impl_.get_service().close(impl_.get_implementation(), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


native_handle_type native_handle()
{
return impl_.get_service().native_handle(impl_.get_implementation());
}


native_handle_type release()
{
return impl_.get_service().release(impl_.get_implementation());
}


void cancel()
{
boost::system::error_code ec;
impl_.get_service().cancel(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "cancel");
}


BOOST_ASIO_SYNC_OP_VOID cancel(boost::system::error_code& ec)
{
impl_.get_service().cancel(impl_.get_implementation(), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename IoControlCommand>
void io_control(IoControlCommand& command)
{
boost::system::error_code ec;
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
boost::asio::detail::throw_error(ec, "io_control");
}


template <typename IoControlCommand>
BOOST_ASIO_SYNC_OP_VOID io_control(IoControlCommand& command,
boost::system::error_code& ec)
{
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool non_blocking() const
{
return impl_.get_service().non_blocking(impl_.get_implementation());
}


void non_blocking(bool mode)
{
boost::system::error_code ec;
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
boost::asio::detail::throw_error(ec, "non_blocking");
}


BOOST_ASIO_SYNC_OP_VOID non_blocking(
bool mode, boost::system::error_code& ec)
{
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool native_non_blocking() const
{
return impl_.get_service().native_non_blocking(
impl_.get_implementation());
}


void native_non_blocking(bool mode)
{
boost::system::error_code ec;
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
boost::asio::detail::throw_error(ec, "native_non_blocking");
}


BOOST_ASIO_SYNC_OP_VOID native_non_blocking(
bool mode, boost::system::error_code& ec)
{
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void wait(wait_type w)
{
boost::system::error_code ec;
impl_.get_service().wait(impl_.get_implementation(), w, ec);
boost::asio::detail::throw_error(ec, "wait");
}


BOOST_ASIO_SYNC_OP_VOID wait(wait_type w, boost::system::error_code& ec)
{
impl_.get_service().wait(impl_.get_implementation(), w, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code))
WaitHandler BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler,
void (boost::system::error_code))
async_wait(wait_type w,
BOOST_ASIO_MOVE_ARG(WaitHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WaitHandler, void (boost::system::error_code)>(
initiate_async_wait(this), handler, w);
}

protected:

~basic_descriptor()
{
}

detail::io_object_impl<detail::reactive_descriptor_service, Executor> impl_;

private:
basic_descriptor(const basic_descriptor&) BOOST_ASIO_DELETED;
basic_descriptor& operator=(const basic_descriptor&) BOOST_ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_descriptor* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WaitHandler>
void operator()(BOOST_ASIO_MOVE_ARG(WaitHandler) handler, wait_type w) const
{
BOOST_ASIO_WAIT_HANDLER_CHECK(WaitHandler, handler) type_check;

detail::non_const_lvalue<WaitHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(), w,
handler2.value, self_->impl_.get_executor());
}

private:
basic_descriptor* self_;
};
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
