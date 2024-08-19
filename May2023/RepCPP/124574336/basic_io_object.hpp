
#ifndef BOOST_ASIO_BASIC_IO_OBJECT_HPP
#define BOOST_ASIO_BASIC_IO_OBJECT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/io_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_HAS_MOVE)
namespace detail
{
template <typename IoObjectService>
class service_has_move
{
private:
typedef IoObjectService service_type;
typedef typename service_type::implementation_type implementation_type;

template <typename T, typename U>
static auto asio_service_has_move_eval(T* t, U* u)
-> decltype(t->move_construct(*u, *u), char());
static char (&asio_service_has_move_eval(...))[2];

public:
static const bool value =
sizeof(asio_service_has_move_eval(
static_cast<service_type*>(0),
static_cast<implementation_type*>(0))) == 1;
};
}
#endif 


#if !defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
template <typename IoObjectService>
#else
template <typename IoObjectService,
bool Movable = detail::service_has_move<IoObjectService>::value>
#endif
class basic_io_object
{
public:
typedef IoObjectService service_type;

typedef typename service_type::implementation_type implementation_type;

#if !defined(BOOST_ASIO_NO_DEPRECATED)

boost::asio::io_context& get_io_context()
{
return service_.get_io_context();
}


boost::asio::io_context& get_io_service()
{
return service_.get_io_context();
}
#endif 

typedef boost::asio::io_context::executor_type executor_type;

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return service_.get_io_context().get_executor();
}

protected:

explicit basic_io_object(boost::asio::io_context& io_context)
: service_(boost::asio::use_service<IoObjectService>(io_context))
{
service_.construct(implementation_);
}

#if defined(GENERATING_DOCUMENTATION)

basic_io_object(basic_io_object&& other);


basic_io_object& operator=(basic_io_object&& other);

template <typename IoObjectService1>
basic_io_object(IoObjectService1& other_service,
typename IoObjectService1::implementation_type& other_implementation);
#endif 


~basic_io_object()
{
service_.destroy(implementation_);
}

service_type& get_service()
{
return service_;
}

const service_type& get_service() const
{
return service_;
}

implementation_type& get_implementation()
{
return implementation_;
}

const implementation_type& get_implementation() const
{
return implementation_;
}

private:
basic_io_object(const basic_io_object&);
basic_io_object& operator=(const basic_io_object&);

service_type& service_;

implementation_type implementation_;
};

#if defined(BOOST_ASIO_HAS_MOVE)
template <typename IoObjectService>
class basic_io_object<IoObjectService, true>
{
public:
typedef IoObjectService service_type;
typedef typename service_type::implementation_type implementation_type;

#if !defined(BOOST_ASIO_NO_DEPRECATED)
boost::asio::io_context& get_io_context()
{
return service_->get_io_context();
}

boost::asio::io_context& get_io_service()
{
return service_->get_io_context();
}
#endif 

typedef boost::asio::io_context::executor_type executor_type;

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return service_->get_io_context().get_executor();
}

protected:
explicit basic_io_object(boost::asio::io_context& io_context)
: service_(&boost::asio::use_service<IoObjectService>(io_context))
{
service_->construct(implementation_);
}

basic_io_object(basic_io_object&& other)
: service_(&other.get_service())
{
service_->move_construct(implementation_, other.implementation_);
}

template <typename IoObjectService1>
basic_io_object(IoObjectService1& other_service,
typename IoObjectService1::implementation_type& other_implementation)
: service_(&boost::asio::use_service<IoObjectService>(
other_service.get_io_context()))
{
service_->converting_move_construct(implementation_,
other_service, other_implementation);
}

~basic_io_object()
{
service_->destroy(implementation_);
}

basic_io_object& operator=(basic_io_object&& other)
{
service_->move_assign(implementation_,
*other.service_, other.implementation_);
service_ = other.service_;
return *this;
}

service_type& get_service()
{
return *service_;
}

const service_type& get_service() const
{
return *service_;
}

implementation_type& get_implementation()
{
return implementation_;
}

const implementation_type& get_implementation() const
{
return implementation_;
}

private:
basic_io_object(const basic_io_object&);
void operator=(const basic_io_object&);

IoObjectService* service_;
implementation_type implementation_;
};
#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
