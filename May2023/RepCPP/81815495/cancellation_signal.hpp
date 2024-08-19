
#ifndef ASIO_CANCELLATION_SIGNAL_HPP
#define ASIO_CANCELLATION_SIGNAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cassert>
#include <new>
#include <utility>
#include "asio/cancellation_type.hpp"
#include "asio/detail/cstddef.hpp"
#include "asio/detail/thread_context.hpp"
#include "asio/detail/thread_info_base.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class cancellation_handler_base
{
public:
virtual void call(cancellation_type_t) = 0;
virtual std::pair<void*, std::size_t> destroy() ASIO_NOEXCEPT = 0;

protected:
~cancellation_handler_base() {}
};

template <typename Handler>
class cancellation_handler
: public cancellation_handler_base
{
public:
#if defined(ASIO_HAS_VARIADIC_TEMPLATES)
template <typename... Args>
cancellation_handler(std::size_t size, ASIO_MOVE_ARG(Args)... args)
: handler_(ASIO_MOVE_CAST(Args)(args)...),
size_(size)
{
}
#else 
cancellation_handler(std::size_t size)
: handler_(),
size_(size)
{
}

#define ASIO_PRIVATE_HANDLER_CTOR_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
cancellation_handler(std::size_t size, ASIO_VARIADIC_MOVE_PARAMS(n)) \
: handler_(ASIO_VARIADIC_MOVE_ARGS(n)), \
size_(size) \
{ \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_HANDLER_CTOR_DEF)
#undef ASIO_PRIVATE_HANDLER_CTOR_DEF
#endif 

void call(cancellation_type_t type)
{
handler_(type);
}

std::pair<void*, std::size_t> destroy() ASIO_NOEXCEPT
{
std::pair<void*, std::size_t> mem(this, size_);
this->cancellation_handler::~cancellation_handler();
return mem;
}

Handler& handler() ASIO_NOEXCEPT
{
return handler_;
}

private:
~cancellation_handler()
{
}

Handler handler_;
std::size_t size_;
};

} 

class cancellation_slot;

class cancellation_signal
{
public:
ASIO_CONSTEXPR cancellation_signal()
: handler_(0)
{
}

~cancellation_signal()
{
if (handler_)
{
std::pair<void*, std::size_t> mem = handler_->destroy();
detail::thread_info_base::deallocate(
detail::thread_info_base::cancellation_signal_tag(),
detail::thread_context::top_of_thread_call_stack(),
mem.first, mem.second);
}
}

void emit(cancellation_type_t type)
{
if (handler_)
handler_->call(type);
}


cancellation_slot slot() ASIO_NOEXCEPT;

private:
cancellation_signal(const cancellation_signal&) ASIO_DELETED;
cancellation_signal& operator=(const cancellation_signal&) ASIO_DELETED;

detail::cancellation_handler_base* handler_;
};

class cancellation_slot
{
public:
ASIO_CONSTEXPR cancellation_slot()
: handler_(0)
{
}

#if defined(ASIO_HAS_VARIADIC_TEMPLATES) \
|| defined(GENERATING_DOCUMENTATION)

template <typename CancellationHandler, typename... Args>
CancellationHandler& emplace(ASIO_MOVE_ARG(Args)... args)
{
typedef detail::cancellation_handler<CancellationHandler>
cancellation_handler_type;
auto_delete_helper del = { prepare_memory(
sizeof(cancellation_handler_type),
ASIO_ALIGNOF(CancellationHandler)) };
cancellation_handler_type* handler_obj =
new (del.mem.first) cancellation_handler_type(
del.mem.second, ASIO_MOVE_CAST(Args)(args)...);
del.mem.first = 0;
*handler_ = handler_obj;
return handler_obj->handler();
}
#else 
template <typename CancellationHandler>
CancellationHandler& emplace()
{
typedef detail::cancellation_handler<CancellationHandler>
cancellation_handler_type;
auto_delete_helper del = { prepare_memory(
sizeof(cancellation_handler_type),
ASIO_ALIGNOF(CancellationHandler)) };
cancellation_handler_type* handler_obj =
new (del.mem.first) cancellation_handler_type(del.mem.second);
del.mem.first = 0;
*handler_ = handler_obj;
return handler_obj->handler();
}

#define ASIO_PRIVATE_HANDLER_EMPLACE_DEF(n) \
template <typename CancellationHandler, ASIO_VARIADIC_TPARAMS(n)> \
CancellationHandler& emplace(ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
typedef detail::cancellation_handler<CancellationHandler> \
cancellation_handler_type; \
auto_delete_helper del = { prepare_memory( \
sizeof(cancellation_handler_type), \
ASIO_ALIGNOF(CancellationHandler)) }; \
cancellation_handler_type* handler_obj = \
new (del.mem.first) cancellation_handler_type( \
del.mem.second, ASIO_VARIADIC_MOVE_ARGS(n)); \
del.mem.first = 0; \
*handler_ = handler_obj; \
return handler_obj->handler(); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_HANDLER_EMPLACE_DEF)
#undef ASIO_PRIVATE_HANDLER_EMPLACE_DEF
#endif 


template <typename CancellationHandler>
typename decay<CancellationHandler>::type& assign(
ASIO_MOVE_ARG(CancellationHandler) handler)
{
return this->emplace<typename decay<CancellationHandler>::type>(
ASIO_MOVE_CAST(CancellationHandler)(handler));
}


void clear()
{
if (handler_ != 0 && *handler_ != 0)
{
std::pair<void*, std::size_t> mem = (*handler_)->destroy();
detail::thread_info_base::deallocate(
detail::thread_info_base::cancellation_signal_tag(),
detail::thread_context::top_of_thread_call_stack(),
mem.first, mem.second);
*handler_ = 0;
}
}

ASIO_CONSTEXPR bool is_connected() const ASIO_NOEXCEPT
{
return handler_ != 0;
}

ASIO_CONSTEXPR bool has_handler() const ASIO_NOEXCEPT
{
return handler_ != 0 && *handler_ != 0;
}

friend ASIO_CONSTEXPR bool operator==(const cancellation_slot& lhs,
const cancellation_slot& rhs) ASIO_NOEXCEPT
{
return lhs.handler_ == rhs.handler_;
}

friend ASIO_CONSTEXPR bool operator!=(const cancellation_slot& lhs,
const cancellation_slot& rhs) ASIO_NOEXCEPT
{
return lhs.handler_ != rhs.handler_;
}

private:
friend class cancellation_signal;

ASIO_CONSTEXPR cancellation_slot(int,
detail::cancellation_handler_base** handler)
: handler_(handler)
{
}

std::pair<void*, std::size_t> prepare_memory(
std::size_t size, std::size_t align)
{
assert(handler_);
std::pair<void*, std::size_t> mem;
if (*handler_)
{
mem = (*handler_)->destroy();
*handler_ = 0;
}
if (size > mem.second
|| reinterpret_cast<std::size_t>(mem.first) % align != 0)
{
if (mem.first)
{
detail::thread_info_base::deallocate(
detail::thread_info_base::cancellation_signal_tag(),
detail::thread_context::top_of_thread_call_stack(),
mem.first, mem.second);
}
mem.first = detail::thread_info_base::allocate(
detail::thread_info_base::cancellation_signal_tag(),
detail::thread_context::top_of_thread_call_stack(),
size, align);
mem.second = size;
}
return mem;
}

struct auto_delete_helper
{
std::pair<void*, std::size_t> mem;

~auto_delete_helper()
{
if (mem.first)
{
detail::thread_info_base::deallocate(
detail::thread_info_base::cancellation_signal_tag(),
detail::thread_context::top_of_thread_call_stack(),
mem.first, mem.second);
}
}
};

detail::cancellation_handler_base** handler_;
};

inline cancellation_slot cancellation_signal::slot() ASIO_NOEXCEPT
{
return cancellation_slot(0, &handler_);
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
