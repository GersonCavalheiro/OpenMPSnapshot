
#ifndef BOOST_ASIO_IP_BASIC_RESOLVER_HPP
#define BOOST_ASIO_IP_BASIC_RESOLVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <string>
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/async_result.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/io_object_impl.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/string_view.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/ip/basic_resolver_iterator.hpp>
#include <boost/asio/ip/basic_resolver_query.hpp>
#include <boost/asio/ip/basic_resolver_results.hpp>
#include <boost/asio/ip/resolver_base.hpp>
#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
# include <boost/asio/detail/winrt_resolver_service.hpp>
#else
# include <boost/asio/detail/resolver_service.hpp>
#endif

#if defined(BOOST_ASIO_HAS_MOVE)
# include <utility>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {

#if !defined(BOOST_ASIO_IP_BASIC_RESOLVER_FWD_DECL)
#define BOOST_ASIO_IP_BASIC_RESOLVER_FWD_DECL

template <typename InternetProtocol, typename Executor = any_io_executor>
class basic_resolver;

#endif 


template <typename InternetProtocol, typename Executor>
class basic_resolver
: public resolver_base
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_resolver<InternetProtocol, Executor1> other;
};

typedef InternetProtocol protocol_type;

typedef typename InternetProtocol::endpoint endpoint_type;

#if !defined(BOOST_ASIO_NO_DEPRECATED)
typedef basic_resolver_query<InternetProtocol> query;

typedef basic_resolver_iterator<InternetProtocol> iterator;
#endif 

typedef basic_resolver_results<InternetProtocol> results_type;


explicit basic_resolver(const executor_type& ex)
: impl_(ex)
{
}


template <typename ExecutionContext>
explicit basic_resolver(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_resolver(basic_resolver&& other)
: impl_(std::move(other.impl_))
{
}

template <typename InternetProtocol1, typename Executor1>
friend class basic_resolver;


template <typename Executor1>
basic_resolver(basic_resolver<InternetProtocol, Executor1>&& other,
typename enable_if<
is_convertible<Executor1, Executor>::value
>::type* = 0)
: impl_(std::move(other.impl_))
{
}


basic_resolver& operator=(basic_resolver&& other)
{
impl_ = std::move(other.impl_);
return *this;
}


template <typename Executor1>
typename enable_if<
is_convertible<Executor1, Executor>::value,
basic_resolver&
>::type operator=(basic_resolver<InternetProtocol, Executor1>&& other)
{
basic_resolver tmp(std::move(other));
impl_ = std::move(tmp.impl_);
return *this;
}
#endif 


~basic_resolver()
{
}

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return impl_.get_executor();
}


void cancel()
{
return impl_.get_service().cancel(impl_.get_implementation());
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

results_type resolve(const query& q)
{
boost::system::error_code ec;
results_type r = impl_.get_service().resolve(
impl_.get_implementation(), q, ec);
boost::asio::detail::throw_error(ec, "resolve");
return r;
}


results_type resolve(const query& q, boost::system::error_code& ec)
{
return impl_.get_service().resolve(impl_.get_implementation(), q, ec);
}
#endif 


results_type resolve(BOOST_ASIO_STRING_VIEW_PARAM host,
BOOST_ASIO_STRING_VIEW_PARAM service)
{
return resolve(host, service, resolver_base::flags());
}


results_type resolve(BOOST_ASIO_STRING_VIEW_PARAM host,
BOOST_ASIO_STRING_VIEW_PARAM service, boost::system::error_code& ec)
{
return resolve(host, service, resolver_base::flags(), ec);
}


results_type resolve(BOOST_ASIO_STRING_VIEW_PARAM host,
BOOST_ASIO_STRING_VIEW_PARAM service, resolver_base::flags resolve_flags)
{
boost::system::error_code ec;
basic_resolver_query<protocol_type> q(static_cast<std::string>(host),
static_cast<std::string>(service), resolve_flags);
results_type r = impl_.get_service().resolve(
impl_.get_implementation(), q, ec);
boost::asio::detail::throw_error(ec, "resolve");
return r;
}


results_type resolve(BOOST_ASIO_STRING_VIEW_PARAM host,
BOOST_ASIO_STRING_VIEW_PARAM service, resolver_base::flags resolve_flags,
boost::system::error_code& ec)
{
basic_resolver_query<protocol_type> q(static_cast<std::string>(host),
static_cast<std::string>(service), resolve_flags);
return impl_.get_service().resolve(impl_.get_implementation(), q, ec);
}


results_type resolve(const protocol_type& protocol,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service)
{
return resolve(protocol, host, service, resolver_base::flags());
}


results_type resolve(const protocol_type& protocol,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service,
boost::system::error_code& ec)
{
return resolve(protocol, host, service, resolver_base::flags(), ec);
}


results_type resolve(const protocol_type& protocol,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service,
resolver_base::flags resolve_flags)
{
boost::system::error_code ec;
basic_resolver_query<protocol_type> q(
protocol, static_cast<std::string>(host),
static_cast<std::string>(service), resolve_flags);
results_type r = impl_.get_service().resolve(
impl_.get_implementation(), q, ec);
boost::asio::detail::throw_error(ec, "resolve");
return r;
}


results_type resolve(const protocol_type& protocol,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service,
resolver_base::flags resolve_flags, boost::system::error_code& ec)
{
basic_resolver_query<protocol_type> q(
protocol, static_cast<std::string>(host),
static_cast<std::string>(service), resolve_flags);
return impl_.get_service().resolve(impl_.get_implementation(), q, ec);
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
results_type)) ResolveHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ResolveHandler,
void (boost::system::error_code, results_type))
async_resolve(const query& q,
BOOST_ASIO_MOVE_ARG(ResolveHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return boost::asio::async_initiate<ResolveHandler,
void (boost::system::error_code, results_type)>(
initiate_async_resolve(this), handler, q);
}
#endif 


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
results_type)) ResolveHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ResolveHandler,
void (boost::system::error_code, results_type))
async_resolve(BOOST_ASIO_STRING_VIEW_PARAM host,
BOOST_ASIO_STRING_VIEW_PARAM service,
BOOST_ASIO_MOVE_ARG(ResolveHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_resolve(host, service, resolver_base::flags(),
BOOST_ASIO_MOVE_CAST(ResolveHandler)(handler));
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
results_type)) ResolveHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ResolveHandler,
void (boost::system::error_code, results_type))
async_resolve(BOOST_ASIO_STRING_VIEW_PARAM host,
BOOST_ASIO_STRING_VIEW_PARAM service,
resolver_base::flags resolve_flags,
BOOST_ASIO_MOVE_ARG(ResolveHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
basic_resolver_query<protocol_type> q(static_cast<std::string>(host),
static_cast<std::string>(service), resolve_flags);

return boost::asio::async_initiate<ResolveHandler,
void (boost::system::error_code, results_type)>(
initiate_async_resolve(this), handler, q);
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
results_type)) ResolveHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ResolveHandler,
void (boost::system::error_code, results_type))
async_resolve(const protocol_type& protocol,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service,
BOOST_ASIO_MOVE_ARG(ResolveHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_resolve(protocol, host, service, resolver_base::flags(),
BOOST_ASIO_MOVE_CAST(ResolveHandler)(handler));
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
results_type)) ResolveHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ResolveHandler,
void (boost::system::error_code, results_type))
async_resolve(const protocol_type& protocol,
BOOST_ASIO_STRING_VIEW_PARAM host, BOOST_ASIO_STRING_VIEW_PARAM service,
resolver_base::flags resolve_flags,
BOOST_ASIO_MOVE_ARG(ResolveHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
basic_resolver_query<protocol_type> q(
protocol, static_cast<std::string>(host),
static_cast<std::string>(service), resolve_flags);

return boost::asio::async_initiate<ResolveHandler,
void (boost::system::error_code, results_type)>(
initiate_async_resolve(this), handler, q);
}


results_type resolve(const endpoint_type& e)
{
boost::system::error_code ec;
results_type i = impl_.get_service().resolve(
impl_.get_implementation(), e, ec);
boost::asio::detail::throw_error(ec, "resolve");
return i;
}


results_type resolve(const endpoint_type& e, boost::system::error_code& ec)
{
return impl_.get_service().resolve(impl_.get_implementation(), e, ec);
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
results_type)) ResolveHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(ResolveHandler,
void (boost::system::error_code, results_type))
async_resolve(const endpoint_type& e,
BOOST_ASIO_MOVE_ARG(ResolveHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return boost::asio::async_initiate<ResolveHandler,
void (boost::system::error_code, results_type)>(
initiate_async_resolve(this), handler, e);
}

private:
basic_resolver(const basic_resolver&) BOOST_ASIO_DELETED;
basic_resolver& operator=(const basic_resolver&) BOOST_ASIO_DELETED;

class initiate_async_resolve
{
public:
typedef Executor executor_type;

explicit initiate_async_resolve(basic_resolver* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ResolveHandler, typename Query>
void operator()(BOOST_ASIO_MOVE_ARG(ResolveHandler) handler,
const Query& q) const
{
BOOST_ASIO_RESOLVE_HANDLER_CHECK(
ResolveHandler, handler, results_type) type_check;

boost::asio::detail::non_const_lvalue<ResolveHandler> handler2(handler);
self_->impl_.get_service().async_resolve(
self_->impl_.get_implementation(), q,
handler2.value, self_->impl_.get_executor());
}

private:
basic_resolver* self_;
};

# if defined(BOOST_ASIO_WINDOWS_RUNTIME)
boost::asio::detail::io_object_impl<
boost::asio::detail::winrt_resolver_service<InternetProtocol>,
Executor> impl_;
# else
boost::asio::detail::io_object_impl<
boost::asio::detail::resolver_service<InternetProtocol>,
Executor> impl_;
# endif
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
