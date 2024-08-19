
#ifndef BOOST_ASIO_STRAND_HPP
#define BOOST_ASIO_STRAND_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/strand_executor_service.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/execution/executor.hpp>
#include <boost/asio/is_executor.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
typename enable_if<
conditional<
!is_same<Executor1, strand>::value,
is_convertible<Executor1, Executor>,
false_type
>::type::value
>::type* = 0)
: executor_(e),
impl_(strand::create_implementation(executor_))
{
}

strand(const strand& other) BOOST_ASIO_NOEXCEPT
: executor_(other.executor_),
impl_(other.impl_)
{
}


template <class OtherExecutor>
strand(
const strand<OtherExecutor>& other) BOOST_ASIO_NOEXCEPT
: executor_(other.executor_),
impl_(other.impl_)
{
}

strand& operator=(const strand& other) BOOST_ASIO_NOEXCEPT
{
executor_ = other.executor_;
impl_ = other.impl_;
return *this;
}


template <class OtherExecutor>
strand& operator=(
const strand<OtherExecutor>& other) BOOST_ASIO_NOEXCEPT
{
executor_ = other.executor_;
impl_ = other.impl_;
return *this;
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
strand(strand&& other) BOOST_ASIO_NOEXCEPT
: executor_(BOOST_ASIO_MOVE_CAST(Executor)(other.executor_)),
impl_(BOOST_ASIO_MOVE_CAST(implementation_type)(other.impl_))
{
}


template <class OtherExecutor>
strand(strand<OtherExecutor>&& other) BOOST_ASIO_NOEXCEPT
: executor_(BOOST_ASIO_MOVE_CAST(OtherExecutor)(other.executor_)),
impl_(BOOST_ASIO_MOVE_CAST(implementation_type)(other.impl_))
{
}

strand& operator=(strand&& other) BOOST_ASIO_NOEXCEPT
{
executor_ = BOOST_ASIO_MOVE_CAST(Executor)(other.executor_);
impl_ = BOOST_ASIO_MOVE_CAST(implementation_type)(other.impl_);
return *this;
}


template <class OtherExecutor>
strand& operator=(strand<OtherExecutor>&& other) BOOST_ASIO_NOEXCEPT
{
executor_ = BOOST_ASIO_MOVE_CAST(OtherExecutor)(other.executor_);
impl_ = BOOST_ASIO_MOVE_CAST(implementation_type)(other.impl_);
return *this;
}
#endif 

~strand() BOOST_ASIO_NOEXCEPT
{
}

inner_executor_type get_inner_executor() const BOOST_ASIO_NOEXCEPT
{
return executor_;
}


template <typename Property>
typename enable_if<
can_query<const Executor&, Property>::value,
typename query_result<const Executor&, Property>::type
>::type query(const Property& p) const
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, Property>::value))
{
return boost::asio::query(executor_, p);
}


template <typename Property>
typename enable_if<
can_require<const Executor&, Property>::value,
strand<typename decay<
typename require_result<const Executor&, Property>::type
>::type>
>::type require(const Property& p) const
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_require<const Executor&, Property>::value))
{
return strand<typename decay<
typename require_result<const Executor&, Property>::type
>::type>(boost::asio::require(executor_, p), impl_);
}


template <typename Property>
typename enable_if<
can_prefer<const Executor&, Property>::value,
strand<typename decay<
typename prefer_result<const Executor&, Property>::type
>::type>
>::type prefer(const Property& p) const
BOOST_ASIO_NOEXCEPT_IF((
is_nothrow_prefer<const Executor&, Property>::value))
{
return strand<typename decay<
typename prefer_result<const Executor&, Property>::type
>::type>(boost::asio::prefer(executor_, p), impl_);
}

#if !defined(BOOST_ASIO_NO_TS_EXECUTORS)
execution_context& context() const BOOST_ASIO_NOEXCEPT
{
return executor_.context();
}


void on_work_started() const BOOST_ASIO_NOEXCEPT
{
executor_.on_work_started();
}


void on_work_finished() const BOOST_ASIO_NOEXCEPT
{
executor_.on_work_finished();
}
#endif 


template <typename Function>
typename enable_if<
execution::can_execute<const Executor&, Function>::value
>::type execute(BOOST_ASIO_MOVE_ARG(Function) f) const
{
detail::strand_executor_service::execute(impl_,
executor_, BOOST_ASIO_MOVE_CAST(Function)(f));
}

#if !defined(BOOST_ASIO_NO_TS_EXECUTORS)

template <typename Function, typename Allocator>
void dispatch(BOOST_ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
detail::strand_executor_service::dispatch(impl_,
executor_, BOOST_ASIO_MOVE_CAST(Function)(f), a);
}


template <typename Function, typename Allocator>
void post(BOOST_ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
detail::strand_executor_service::post(impl_,
executor_, BOOST_ASIO_MOVE_CAST(Function)(f), a);
}


template <typename Function, typename Allocator>
void defer(BOOST_ASIO_MOVE_ARG(Function) f, const Allocator& a) const
{
detail::strand_executor_service::defer(impl_,
executor_, BOOST_ASIO_MOVE_CAST(Function)(f), a);
}
#endif 


bool running_in_this_thread() const BOOST_ASIO_NOEXCEPT
{
return detail::strand_executor_service::running_in_this_thread(impl_);
}


friend bool operator==(const strand& a, const strand& b) BOOST_ASIO_NOEXCEPT
{
return a.impl_ == b.impl_;
}


friend bool operator!=(const strand& a, const strand& b) BOOST_ASIO_NOEXCEPT
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
typename enable_if<
can_query<InnerExecutor, execution::context_t>::value
>::type* = 0)
{
return use_service<detail::strand_executor_service>(
boost::asio::query(ex, execution::context)).create_implementation();
}

template <typename InnerExecutor>
static implementation_type create_implementation(const InnerExecutor& ex,
typename enable_if<
!can_query<InnerExecutor, execution::context_t>::value
>::type* = 0)
{
return use_service<detail::strand_executor_service>(
ex.context()).create_implementation();
}

strand(const Executor& ex, const implementation_type& impl)
: executor_(ex),
impl_(impl)
{
}

Executor executor_;
implementation_type impl_;
};




template <typename Executor>
inline strand<Executor> make_strand(const Executor& ex,
typename enable_if<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type* = 0)
{
return strand<Executor>(ex);
}

template <typename ExecutionContext>
inline strand<typename ExecutionContext::executor_type>
make_strand(ExecutionContext& ctx,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
return strand<typename ExecutionContext::executor_type>(ctx.get_executor());
}



#if !defined(GENERATING_DOCUMENTATION)

namespace traits {

#if !defined(BOOST_ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <typename Executor>
struct equality_comparable<strand<Executor> >
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename Executor, typename Function>
struct execute_member<strand<Executor>, Function,
typename enable_if<
execution::can_execute<const Executor&, Function>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

template <typename Executor, typename Property>
struct query_member<strand<Executor>, Property,
typename enable_if<
can_query<const Executor&, Property>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<Executor, Property>::value));
typedef typename query_result<Executor, Property>::type result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

template <typename Executor, typename Property>
struct require_member<strand<Executor>, Property,
typename enable_if<
can_require<const Executor&, Property>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_require<Executor, Property>::value));
typedef strand<typename decay<
typename require_result<Executor, Property>::type
>::type> result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

template <typename Executor, typename Property>
struct prefer_member<strand<Executor>, Property,
typename enable_if<
can_prefer<const Executor&, Property>::value
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_prefer<Executor, Property>::value));
typedef strand<typename decay<
typename prefer_result<Executor, Property>::type
>::type> result_type;
};

#endif 

} 

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
# if defined(BOOST_ASIO_IO_CONTEXT_HPP)
#  include <boost/asio/io_context_strand.hpp>
# endif 
#endif 

#endif 
