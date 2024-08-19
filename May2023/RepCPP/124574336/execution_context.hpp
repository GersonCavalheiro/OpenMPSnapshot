
#ifndef BOOST_ASIO_IMPL_EXECUTION_CONTEXT_HPP
#define BOOST_ASIO_IMPL_EXECUTION_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/scoped_ptr.hpp>
#include <boost/asio/detail/service_registry.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if !defined(GENERATING_DOCUMENTATION)

template <typename Service>
inline Service& use_service(execution_context& e)
{
(void)static_cast<execution_context::service*>(static_cast<Service*>(0));

return e.service_registry_->template use_service<Service>();
}

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Service, typename... Args>
Service& make_service(execution_context& e, BOOST_ASIO_MOVE_ARG(Args)... args)
{
detail::scoped_ptr<Service> svc(
new Service(e, BOOST_ASIO_MOVE_CAST(Args)(args)...));
e.service_registry_->template add_service<Service>(svc.get());
Service& result = *svc;
svc.release();
return result;
}

#else 

template <typename Service>
Service& make_service(execution_context& e)
{
detail::scoped_ptr<Service> svc(new Service(e));
e.service_registry_->template add_service<Service>(svc.get());
Service& result = *svc;
svc.release();
return result;
}

#define BOOST_ASIO_PRIVATE_MAKE_SERVICE_DEF(n) \
template <typename Service, BOOST_ASIO_VARIADIC_TPARAMS(n)> \
Service& make_service(execution_context& e, \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
detail::scoped_ptr<Service> svc( \
new Service(e, BOOST_ASIO_VARIADIC_MOVE_ARGS(n))); \
e.service_registry_->template add_service<Service>(svc.get()); \
Service& result = *svc; \
svc.release(); \
return result; \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_MAKE_SERVICE_DEF)
#undef BOOST_ASIO_PRIVATE_MAKE_SERVICE_DEF

#endif 

template <typename Service>
inline void add_service(execution_context& e, Service* svc)
{
(void)static_cast<execution_context::service*>(static_cast<Service*>(0));

e.service_registry_->template add_service<Service>(svc);
}

template <typename Service>
inline bool has_service(execution_context& e)
{
(void)static_cast<execution_context::service*>(static_cast<Service*>(0));

return e.service_registry_->template has_service<Service>();
}

#endif 

inline execution_context& execution_context::service::context()
{
return owner_;
}

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
