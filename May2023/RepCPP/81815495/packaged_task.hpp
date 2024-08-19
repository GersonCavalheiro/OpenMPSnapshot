
#ifndef ASIO_PACKAGED_TASK_HPP
#define ASIO_PACKAGED_TASK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/future.hpp"

#if defined(ASIO_HAS_STD_FUTURE_CLASS) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/async_result.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
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

#define ASIO_PRIVATE_ASYNC_RESULT_DEF(n) \
template <typename Result, \
ASIO_VARIADIC_TPARAMS(n), typename Signature> \
class async_result< \
std::packaged_task<Result(ASIO_VARIADIC_TARGS(n))>, Signature> \
{ \
public: \
typedef std::packaged_task< \
Result(ASIO_VARIADIC_TARGS(n))> \
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

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_ASYNC_RESULT_DEF)
#undef ASIO_PRIVATE_ASYNC_RESULT_DEF

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
