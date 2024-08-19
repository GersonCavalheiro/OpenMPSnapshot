
#ifndef BOOST_ASIO_DETAIL_EXECUTOR_FUNCTION_HPP
#define BOOST_ASIO_DETAIL_EXECUTOR_FUNCTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/memory.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

#if defined(BOOST_ASIO_HAS_MOVE)

class executor_function
{
public:
template <typename F, typename Alloc>
explicit executor_function(F f, const Alloc& a)
{
typedef impl<F, Alloc> impl_type;
typename impl_type::ptr p = {
detail::addressof(a), impl_type::ptr::allocate(a), 0 };
impl_ = new (p.v) impl_type(BOOST_ASIO_MOVE_CAST(F)(f), a);
p.v = 0;
}

executor_function(executor_function&& other) BOOST_ASIO_NOEXCEPT
: impl_(other.impl_)
{
other.impl_ = 0;
}

~executor_function()
{
if (impl_)
impl_->complete_(impl_, false);
}

void operator()()
{
if (impl_)
{
impl_base* i = impl_;
impl_ = 0;
i->complete_(i, true);
}
}

private:
struct impl_base
{
void (*complete_)(impl_base*, bool);
};

template <typename Function, typename Alloc>
struct impl : impl_base
{
BOOST_ASIO_DEFINE_TAGGED_HANDLER_ALLOCATOR_PTR(
thread_info_base::executor_function_tag, impl);

template <typename F>
impl(BOOST_ASIO_MOVE_ARG(F) f, const Alloc& a)
: function_(BOOST_ASIO_MOVE_CAST(F)(f)),
allocator_(a)
{
complete_ = &executor_function::complete<Function, Alloc>;
}

Function function_;
Alloc allocator_;
};

template <typename Function, typename Alloc>
static void complete(impl_base* base, bool call)
{
impl<Function, Alloc>* i(static_cast<impl<Function, Alloc>*>(base));
Alloc allocator(i->allocator_);
typename impl<Function, Alloc>::ptr p = {
detail::addressof(allocator), i, i };

Function function(BOOST_ASIO_MOVE_CAST(Function)(i->function_));
p.reset();

if (call)
{
boost_asio_handler_invoke_helpers::invoke(function, function);
}
}

impl_base* impl_;
};

#else 

class executor_function
{
public:
template <typename F, typename Alloc>
explicit executor_function(const F& f, const Alloc&)
: impl_(new impl<typename decay<F>::type>(f))
{
}

void operator()()
{
impl_->complete_(impl_.get());
}

private:
struct impl_base
{
void (*complete_)(impl_base*);
};

template <typename F>
struct impl : impl_base
{
impl(const F& f)
: function_(f)
{
complete_ = &executor_function::complete<F>;
}

F function_;
};

template <typename F>
static void complete(impl_base* i)
{
static_cast<impl<F>*>(i)->function_();
}

shared_ptr<impl_base> impl_;
};

#endif 

class executor_function_view
{
public:
template <typename F>
explicit executor_function_view(F& f) BOOST_ASIO_NOEXCEPT
: complete_(&executor_function_view::complete<F>),
function_(&f)
{
}

void operator()()
{
complete_(function_);
}

private:
template <typename F>
static void complete(void* f)
{
(*static_cast<F*>(f))();
}

void (*complete_)(void*);
void* function_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
