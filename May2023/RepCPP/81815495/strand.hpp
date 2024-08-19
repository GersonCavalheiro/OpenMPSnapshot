
#ifndef ASIO_STRAND_HPP
#define ASIO_STRAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/strand_executor_service.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/blocking.hpp"
#include "asio/execution/executor.hpp"
#include "asio/is_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename Executor>
class strand
{
public:
typedef Executor inner_executor_type;


strand()
: executor_(),
impl_(strand::create_implementation(executor_))
{
}

template <typename Executor1>
explicit strand(const Executor1& e,
typename constraint<
conditional<
!is_same<Executor1, strand>::value,
is_convertible<Executor1, Executor>,
false_type
>::type::value
>::type = 0)
: executor_(e),
impl_(strand::create_implementation(executor_))
{
}

strand(const strand& other) ASIO_NOEXCEPT
: executor_(other.executor_),
impl_(other.impl_)
{
}


template <class OtherExecutor>
strand(
const strand<OtherExecutor>& other) ASIO_NOEXCEPT
: executor_(other.executor_),
impl_(other.impl_)
{
}

strand& operator=(const strand& other) ASIO_NOEXCEPT
{
executor_ = other.executor_;
impl_ = other.impl_;
return *this;
}


template <class OtherExecutor>
strand& operator=(
const strand<OtherExecutor>& other) ASIO_NOEXCEPT
{
executor_ = other.executor_;
impl_ = other.impl_;
return *this;
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
strand(strand&& other) ASIO_NOEXCEPT
: executor_(ASIO_MOVE_CAST(Executor)(other.executor_)),
impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_))
{
}


template <class OtherExecutor>
strand(strand<OtherExecutor>&& other) ASIO_NOEXCEPT
: executor_(ASIO_MOVE_CAST(OtherExecutor)(other.executor_)),
impl_(ASIO_MOVE_CAST(implementation_type)(other.impl_))
{
}

strand& operator=(strand&& other) ASIO_NOEXCEPT
{
executor_ = ASIO_MOVE_CAST(Executor)(other.executor_);
impl_ = ASIO_MOVE_CAST(implementation_type)(other.impl_);
return *this;
}


template <class OtherExecutor>
strand& operator=(strand<OtherExecutor>&& other) ASIO_NOEXCEPT
{
executor_ = ASIO_MOVE_CAST(OtherExecutor)(other.executor_);
impl_ = ASIO_MOVE_CAST(implementation_type)(other.impl_);
return *this;
}
#endif 

~strand() ASIO_NOEXCEPT
{
}

inner_executor_type get_inner_executor() const ASIO_NOEXCEPT
{
return executor_;
}


template <typename Property>
typename constraint<
can_query<const Executor&, Property>::value,
typename conditional<
is_convertible<Property, execution::blocking_t>::value,
execution::blocking_t,
typename query_result<const Executor&, Property>::type
>::type
>::type query(const Property& p) const
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, Property>::value))
{
return this->query_helper(
is_convertible<Property, execution::blocking_t>(), p);
}


template <typename Property>
typename constraint<
can_require<const Executor&, Property>::value
&& !is_convertible<Property, execution::blocking_t::always_t>::value,
strand<typename decay<
typename require_result<const Executor&, Property>::type
>::type>
>::type require(const Property& p) const
ASIO_NOEXCEPT_IF((
is_nothrow_require<const Executor&, Property>::value))
{
return strand<typename decay<
typename require_result<const Executor&, Property>::type
>::type>(asio::require(executor_, p), impl_);
}


template <typename Property>
typename constraint<
can_prefer<const Executor&, Property>::value
&& !is_convertible<Property, execution::blocking_t::always_t>::value,
strand<typename decay<
typename prefer_result<const Executor&, Property>::type
>::type>
>::type prefer(const Property& p) const
ASIO_NOEXCEPT_IF((
is_nothrow_prefer<const Executor&, Property>::value))
{
return strand<typename decay<
typename prefer_result<const Executor&, Property>::type
>::type>(asio::prefer(executor_, p), impl_);
}

#if !defined(ASIO_NO_TS_EXECUTORS)
execution_context& context() const ASIO_NOEXCEPT
{
return executor_.context();
}


void on_work_started() const ASIO_NOEXCEPT
{
executor_.on_work_started();
}


void on_work_finished() const ASIO_NOEXCEPT
{
executor_.on_work_finished();
}
#endif 


template <typename Function>
typename constraint<
execution::can_execute<const Executor&, Function>::value,
void
>::type execute(ASIO_MOVE_ARG(Function) f) const
{
detail::strand_executor_service::execute(impl_,
executor_, ASIO_MOVE_CAST(Function)(f));
}

#if !defined(ASIO_NO_TS_EXECUTORS)

template <typename Function, typename Allocator>
void dispatch(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
detail::strand_executor_service::dispatch(impl_,
executor_, ASIO_MOVE_CAST(Function)(f), a);
}


template <typename Function, typename Allocator>
void post(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
detail::strand_executor_service::post(impl_,
executor_, ASIO_MOVE_CAST(Function)(f), a);
}


template <typename Function, typename Allocator>
void defer(ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
detail::strand_executor_service::defer(impl_,
executor_, ASIO_MOVE_CAST(Function)(f), a);
}
#endif 


bool running_in_this_thread() const ASIO_NOEXCEPT
{
return detail::strand_executor_service::running_in_this_thread(impl_);
}


friend bool operator==(const strand& a, const strand& b) ASIO_NOEXCEPT
{
return a.impl_ == b.impl_;
}


friend bool operator!=(const strand& a, const strand& b) ASIO_NOEXCEPT
{
return a.impl_ != b.impl_;
}

#if defined(GENERATING_DOCUMENTATION)
private:
#endif 
typedef detail::strand_executor_service::implementation_type
implementation_type;

template <typename InnerExecutor>
static implementation_type create_implementation(const InnerExecutor& ex,
typename constraint<
can_query<InnerExecutor, execution::context_t>::value
>::type = 0)
{
return use_service<detail::strand_executor_service>(
asio::query(ex, execution::context)).create_implementation();
}

template <typename InnerExecutor>
static implementation_type create_implementation(const InnerExecutor& ex,
typename constraint<
!can_query<InnerExecutor, execution::context_t>::value
>::type = 0)
{
return use_service<detail::strand_executor_service>(
ex.context()).create_implementation();
}

strand(const Executor& ex, const implementation_type& impl)
: executor_(ex),
impl_(impl)
{
}

template <typename Property>
typename query_result<const Executor&, Property>::type query_helper(
false_type, const Property& property) const
{
return asio::query(executor_, property);
}

template <typename Property>
execution::blocking_t query_helper(true_type, const Property& property) const
{
execution::blocking_t result = asio::query(executor_, property);
return result == execution::blocking.always
? execution::blocking.possibly : result;
}

Executor executor_;
implementation_type impl_;
};




template <typename Executor>
inline strand<Executor> make_strand(const Executor& ex,
typename constraint<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type = 0)
{
return strand<Executor>(ex);
}

template <typename ExecutionContext>
inline strand<typename ExecutionContext::executor_type>
make_strand(ExecutionContext& ctx,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
{
return strand<typename ExecutionContext::executor_type>(ctx.get_executor());
}



#if !defined(GENERATING_DOCUMENTATION)

namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <typename Executor>
struct equality_comparable<strand<Executor> >
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename Executor, typename Function>
struct execute_member<strand<Executor>, Function,
typename enable_if<
execution::can_execute<const Executor&, Function>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

template <typename Executor, typename Property>
struct query_member<strand<Executor>, Property,
typename enable_if<
can_query<const Executor&, Property>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<Executor, Property>::value));
typedef typename conditional<
is_convertible<Property, execution::blocking_t>::value,
execution::blocking_t, typename query_result<Executor, Property>::type
>::type result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

template <typename Executor, typename Property>
struct require_member<strand<Executor>, Property,
typename enable_if<
can_require<const Executor&, Property>::value
&& !is_convertible<Property, execution::blocking_t::always_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_require<Executor, Property>::value));
typedef strand<typename decay<
typename require_result<Executor, Property>::type
>::type> result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

template <typename Executor, typename Property>
struct prefer_member<strand<Executor>, Property,
typename enable_if<
can_prefer<const Executor&, Property>::value
&& !is_convertible<Property, execution::blocking_t::always_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_prefer<Executor, Property>::value));
typedef strand<typename decay<
typename prefer_result<Executor, Property>::type
>::type> result_type;
};

#endif 

} 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#if !defined(ASIO_NO_EXTENSIONS)
# if defined(ASIO_IO_CONTEXT_HPP)
#  include "asio/io_context_strand.hpp"
# endif 
#endif 

#endif 
