
#ifndef BOOST_ASIO_IMPL_AWAITABLE_HPP
#define BOOST_ASIO_IMPL_AWAITABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <exception>
#include <new>
#include <tuple>
#include <utility>
#include <boost/asio/detail/thread_context.hpp>
#include <boost/asio/detail/thread_info_base.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/post.hpp>
#include <boost/system/system_error.hpp>
#include <boost/asio/this_coro.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {


template <typename Executor>
class awaitable_frame_base
{
public:
#if !defined(BOOST_ASIO_DISABLE_AWAITABLE_FRAME_RECYCLING)
void* operator new(std::size_t size)
{
return boost::asio::detail::thread_info_base::allocate(
boost::asio::detail::thread_info_base::awaitable_frame_tag(),
boost::asio::detail::thread_context::thread_call_stack::top(),
size);
}

void operator delete(void* pointer, std::size_t size)
{
boost::asio::detail::thread_info_base::deallocate(
boost::asio::detail::thread_info_base::awaitable_frame_tag(),
boost::asio::detail::thread_context::thread_call_stack::top(),
pointer, size);
}
#endif 

auto initial_suspend() noexcept
{
return suspend_always();
}

auto final_suspend() noexcept
{
struct result
{
awaitable_frame_base* this_;

bool await_ready() const noexcept
{
return false;
}

void await_suspend(coroutine_handle<void>) noexcept
{
this->this_->pop_frame();
}

void await_resume() const noexcept
{
}
};

return result{this};
}

void set_except(std::exception_ptr e) noexcept
{
pending_exception_ = e;
}

void set_error(const boost::system::error_code& ec)
{
this->set_except(std::make_exception_ptr(boost::system::system_error(ec)));
}

void unhandled_exception()
{
set_except(std::current_exception());
}

void rethrow_exception()
{
if (pending_exception_)
{
std::exception_ptr ex = std::exchange(pending_exception_, nullptr);
std::rethrow_exception(ex);
}
}

template <typename T>
auto await_transform(awaitable<T, Executor> a) const
{
return a;
}

auto await_transform(this_coro::executor_t) noexcept
{
struct result
{
awaitable_frame_base* this_;

bool await_ready() const noexcept
{
return true;
}

void await_suspend(coroutine_handle<void>) noexcept
{
}

auto await_resume() const noexcept
{
return this_->attached_thread_->get_executor();
}
};

return result{this};
}

template <typename Function>
auto await_transform(Function f,
typename enable_if<
is_convertible<
typename result_of<Function(awaitable_frame_base*)>::type,
awaitable_thread<Executor>*
>::value
>::type* = 0)
{
struct result
{
Function function_;
awaitable_frame_base* this_;

bool await_ready() const noexcept
{
return false;
}

void await_suspend(coroutine_handle<void>) noexcept
{
function_(this_);
}

void await_resume() const noexcept
{
}
};

return result{std::move(f), this};
}

void attach_thread(awaitable_thread<Executor>* handler) noexcept
{
attached_thread_ = handler;
}

awaitable_thread<Executor>* detach_thread() noexcept
{
return std::exchange(attached_thread_, nullptr);
}

void push_frame(awaitable_frame_base<Executor>* caller) noexcept
{
caller_ = caller;
attached_thread_ = caller_->attached_thread_;
attached_thread_->top_of_stack_ = this;
caller_->attached_thread_ = nullptr;
}

void pop_frame() noexcept
{
if (caller_)
caller_->attached_thread_ = attached_thread_;
attached_thread_->top_of_stack_ = caller_;
attached_thread_ = nullptr;
caller_ = nullptr;
}

void resume()
{
coro_.resume();
}

void destroy()
{
coro_.destroy();
}

protected:
coroutine_handle<void> coro_ = nullptr;
awaitable_thread<Executor>* attached_thread_ = nullptr;
awaitable_frame_base<Executor>* caller_ = nullptr;
std::exception_ptr pending_exception_ = nullptr;
};

template <typename T, typename Executor>
class awaitable_frame
: public awaitable_frame_base<Executor>
{
public:
awaitable_frame() noexcept
{
}

awaitable_frame(awaitable_frame&& other) noexcept
: awaitable_frame_base<Executor>(std::move(other))
{
}

~awaitable_frame()
{
if (has_result_)
static_cast<T*>(static_cast<void*>(result_))->~T();
}

awaitable<T, Executor> get_return_object() noexcept
{
this->coro_ = coroutine_handle<awaitable_frame>::from_promise(*this);
return awaitable<T, Executor>(this);
};

template <typename U>
void return_value(U&& u)
{
new (&result_) T(std::forward<U>(u));
has_result_ = true;
}

template <typename... Us>
void return_values(Us&&... us)
{
this->return_value(std::forward_as_tuple(std::forward<Us>(us)...));
}

T get()
{
this->caller_ = nullptr;
this->rethrow_exception();
return std::move(*static_cast<T*>(static_cast<void*>(result_)));
}

private:
alignas(T) unsigned char result_[sizeof(T)];
bool has_result_ = false;
};

template <typename Executor>
class awaitable_frame<void, Executor>
: public awaitable_frame_base<Executor>
{
public:
awaitable<void, Executor> get_return_object()
{
this->coro_ = coroutine_handle<awaitable_frame>::from_promise(*this);
return awaitable<void, Executor>(this);
};

void return_void()
{
}

void get()
{
this->caller_ = nullptr;
this->rethrow_exception();
}
};

template <typename Executor>
class awaitable_thread
{
public:
typedef Executor executor_type;

awaitable_thread(awaitable<void, Executor> p, const Executor& ex)
: bottom_of_stack_(std::move(p)),
top_of_stack_(bottom_of_stack_.frame_),
executor_(ex)
{
}

awaitable_thread(awaitable_thread&& other) noexcept
: bottom_of_stack_(std::move(other.bottom_of_stack_)),
top_of_stack_(std::exchange(other.top_of_stack_, nullptr)),
executor_(std::move(other.executor_))
{
}

~awaitable_thread()
{
if (bottom_of_stack_.valid())
{
(post)(executor_,
[a = std::move(bottom_of_stack_)]() mutable
{
awaitable<void, Executor>(std::move(a));
});
}
}

executor_type get_executor() const noexcept
{
return executor_;
}

void launch()
{
top_of_stack_->attach_thread(this);
pump();
}

protected:
template <typename> friend class awaitable_frame_base;

void pump()
{
do top_of_stack_->resume(); while (top_of_stack_);
if (bottom_of_stack_.valid())
{
awaitable<void, Executor> a(std::move(bottom_of_stack_));
a.frame_->rethrow_exception();
}
}

awaitable<void, Executor> bottom_of_stack_;
awaitable_frame_base<Executor>* top_of_stack_;
executor_type executor_;
};

} 
} 
} 

#if !defined(GENERATING_DOCUMENTATION)
# if defined(BOOST_ASIO_HAS_STD_COROUTINE)

namespace std {

template <typename T, typename Executor, typename... Args>
struct coroutine_traits<boost::asio::awaitable<T, Executor>, Args...>
{
typedef boost::asio::detail::awaitable_frame<T, Executor> promise_type;
};

} 

# else 

namespace std { namespace experimental {

template <typename T, typename Executor, typename... Args>
struct coroutine_traits<boost::asio::awaitable<T, Executor>, Args...>
{
typedef boost::asio::detail::awaitable_frame<T, Executor> promise_type;
};

}} 

# endif 
#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
