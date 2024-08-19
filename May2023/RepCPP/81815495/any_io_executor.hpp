
#ifndef ASIO_ANY_IO_EXECUTOR_HPP
#define ASIO_ANY_IO_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#if defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)
# include "asio/executor.hpp"
#else 
# include "asio/execution.hpp"
# include "asio/execution_context.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)

typedef executor any_io_executor;

#else 


class any_io_executor :
#if defined(GENERATING_DOCUMENTATION)
public execution::any_executor<...>
#else 
public execution::any_executor<
execution::context_as_t<execution_context&>,
execution::blocking_t::never_t,
execution::prefer_only<execution::blocking_t::possibly_t>,
execution::prefer_only<execution::outstanding_work_t::tracked_t>,
execution::prefer_only<execution::outstanding_work_t::untracked_t>,
execution::prefer_only<execution::relationship_t::fork_t>,
execution::prefer_only<execution::relationship_t::continuation_t>
>
#endif 
{
public:
#if !defined(GENERATING_DOCUMENTATION)
typedef execution::any_executor<
execution::context_as_t<execution_context&>,
execution::blocking_t::never_t,
execution::prefer_only<execution::blocking_t::possibly_t>,
execution::prefer_only<execution::outstanding_work_t::tracked_t>,
execution::prefer_only<execution::outstanding_work_t::untracked_t>,
execution::prefer_only<execution::relationship_t::fork_t>,
execution::prefer_only<execution::relationship_t::continuation_t>
> base_type;

typedef void supportable_properties_type(
execution::context_as_t<execution_context&>,
execution::blocking_t::never_t,
execution::prefer_only<execution::blocking_t::possibly_t>,
execution::prefer_only<execution::outstanding_work_t::tracked_t>,
execution::prefer_only<execution::outstanding_work_t::untracked_t>,
execution::prefer_only<execution::relationship_t::fork_t>,
execution::prefer_only<execution::relationship_t::continuation_t>
);
#endif 

any_io_executor() ASIO_NOEXCEPT
: base_type()
{
}

any_io_executor(nullptr_t) ASIO_NOEXCEPT
: base_type(nullptr_t())
{
}

any_io_executor(const any_io_executor& e) ASIO_NOEXCEPT
: base_type(static_cast<const base_type&>(e))
{
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
any_io_executor(any_io_executor&& e) ASIO_NOEXCEPT
: base_type(static_cast<base_type&&>(e))
{
}
#endif 

#if defined(GENERATING_DOCUMENTATION)
template <class... OtherSupportableProperties>
any_io_executor(execution::any_executor<OtherSupportableProperties...> e);
#else 
template <typename OtherAnyExecutor>
any_io_executor(OtherAnyExecutor e,
typename constraint<
conditional<
!is_same<OtherAnyExecutor, any_io_executor>::value
&& is_base_of<execution::detail::any_executor_base,
OtherAnyExecutor>::value,
typename execution::detail::supportable_properties<
0, supportable_properties_type>::template
is_valid_target<OtherAnyExecutor>,
false_type
>::type::value
>::type = 0)
: base_type(ASIO_MOVE_CAST(OtherAnyExecutor)(e))
{
}
#endif 

#if defined(GENERATING_DOCUMENTATION)
template <ASIO_EXECUTION_EXECUTOR Executor>
any_io_executor(Executor e);
#else 
template <ASIO_EXECUTION_EXECUTOR Executor>
any_io_executor(Executor e,
typename constraint<
conditional<
!is_same<Executor, any_io_executor>::value
&& !is_base_of<execution::detail::any_executor_base,
Executor>::value,
execution::detail::is_valid_target_executor<
Executor, supportable_properties_type>,
false_type
>::type::value
>::type = 0)
: base_type(ASIO_MOVE_CAST(Executor)(e))
{
}
#endif 

any_io_executor& operator=(const any_io_executor& e) ASIO_NOEXCEPT
{
base_type::operator=(static_cast<const base_type&>(e));
return *this;
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
any_io_executor& operator=(any_io_executor&& e) ASIO_NOEXCEPT
{
base_type::operator=(static_cast<base_type&&>(e));
return *this;
}
#endif 

any_io_executor& operator=(nullptr_t)
{
base_type::operator=(nullptr_t());
return *this;
}

~any_io_executor()
{
}

void swap(any_io_executor& other) ASIO_NOEXCEPT
{
static_cast<base_type&>(*this).swap(static_cast<base_type&>(other));
}


template <typename Property>
any_io_executor require(const Property& p,
typename constraint<
traits::require_member<const base_type&, const Property&>::is_valid
>::type = 0) const
{
return static_cast<const base_type&>(*this).require(p);
}


template <typename Property>
any_io_executor prefer(const Property& p,
typename constraint<
traits::prefer_member<const base_type&, const Property&>::is_valid
>::type = 0) const
{
return static_cast<const base_type&>(*this).prefer(p);
}
};

#if !defined(GENERATING_DOCUMENTATION)

namespace traits {

#if !defined(ASIO_HAS_DEDUCED_EQUALITY_COMPARABLE_TRAIT)

template <>
struct equality_comparable<any_io_executor>
{
static const bool is_valid = true;
static const bool is_noexcept = true;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)

template <typename F>
struct execute_member<any_io_executor, F>
{
static const bool is_valid = true;
static const bool is_noexcept = false;
typedef void result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)

template <typename Prop>
struct query_member<any_io_executor, Prop> :
query_member<any_io_executor::base_type, Prop>
{
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_REQUIRE_MEMBER_TRAIT)

template <typename Prop>
struct require_member<any_io_executor, Prop> :
require_member<any_io_executor::base_type, Prop>
{
typedef any_io_executor result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_PREFER_MEMBER_TRAIT)

template <typename Prop>
struct prefer_member<any_io_executor, Prop> :
prefer_member<any_io_executor::base_type, Prop>
{
typedef any_io_executor result_type;
};

#endif 

} 

#endif 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
