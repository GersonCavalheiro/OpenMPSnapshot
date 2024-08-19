
#ifndef BOOST_ASIO_PACKAGED_TASK_HPP
#define BOOST_ASIO_PACKAGED_TASK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/future.hpp>

#if defined(BOOST_ASIO_HAS_STD_FUTURE_CLASS) \
|| defined(GENERATING_DOCUMENTATION)

#include <boost/asio/async_result.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/detail/variadic_templates.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES) \
|| defined(GENERATING_DOCUMENTATION)

template <typename Result, typename... Args, typename Signature>
class async_result<std::packaged_task<Result(Args...)>, Signature>
{
public:
typedef std::packaged_task<Result(Args...)> completion_handler_type;

typedef std::future<Result> return_type;

explicit async_result(completion_handler_type& h)
: future_(h.get_future())
{
}

return_type get()
{
return std::move(future_);
}

private:
return_type future_;
};

#else 

template <typename Result, typename Signature>
struct async_result<std::packaged_task<Result()>, Signature>
{
typedef std::packaged_task<Result()> completion_handler_type;
typedef std::future<Result> return_type;

explicit async_result(completion_handler_type& h)
: future_(h.get_future())
{
}

return_type get()
{
return std::move(future_);
}

private:
return_type future_;
};

#define BOOST_ASIO_PRIVATE_ASYNC_RESULT_DEF(n) \
template <typename Result, \
BOOST_ASIO_VARIADIC_TPARAMS(n), typename Signature> \
class async_result< \
std::packaged_task<Result(BOOST_ASIO_VARIADIC_TARGS(n))>, Signature> \
{ \
public: \
typedef std::packaged_task< \
Result(BOOST_ASIO_VARIADIC_TARGS(n))> \
completion_handler_type; \
\
typedef std::future<Result> return_type; \
\
explicit async_result(completion_handler_type& h) \
: future_(h.get_future()) \
{ \
} \
\
return_type get() \
{ \
return std::move(future_); \
} \
\
private: \
return_type future_; \
}; \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_ASYNC_RESULT_DEF)
#undef BOOST_ASIO_PRIVATE_ASYNC_RESULT_DEF

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
