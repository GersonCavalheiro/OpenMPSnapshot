
#ifndef ASIO_EXECUTION_OUTSTANDING_WORK_HPP
#define ASIO_EXECUTION_OUTSTANDING_WORK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/scheduler.hpp"
#include "asio/execution/sender.hpp"
#include "asio/is_applicable_property.hpp"
#include "asio/query.hpp"
#include "asio/traits/query_free.hpp"
#include "asio/traits/query_member.hpp"
#include "asio/traits/query_static_constexpr_member.hpp"
#include "asio/traits/static_query.hpp"
#include "asio/traits/static_require.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(GENERATING_DOCUMENTATION)

namespace execution {

struct outstanding_work_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = false;

static constexpr bool is_preferable = false;

typedef outstanding_work_t polymorphic_query_result_type;

struct untracked_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = true;

static constexpr bool is_preferable = true;

typedef outstanding_work_t polymorphic_query_result_type;

constexpr untracked_t();


static constexpr outstanding_work_t value();
};

struct tracked_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = true;

static constexpr bool is_preferable = true;

typedef outstanding_work_t polymorphic_query_result_type;

constexpr tracked_t();


static constexpr outstanding_work_t value();
};

static constexpr untracked_t untracked;

static constexpr tracked_t tracked;

constexpr outstanding_work_t();

constexpr outstanding_work_t(untracked_t);

constexpr outstanding_work_t(tracked_t);

friend constexpr bool operator==(
const outstanding_work_t& a, const outstanding_work_t& b) noexcept;

friend constexpr bool operator!=(
const outstanding_work_t& a, const outstanding_work_t& b) noexcept;
};

constexpr outstanding_work_t outstanding_work;

} 

#else 

namespace execution {
namespace detail {
namespace outstanding_work {

template <int I> struct untracked_t;
template <int I> struct tracked_t;

} 

template <int I = 0>
struct outstanding_work_t
{
#if defined(ASIO_HAS_VARIABLE_TEMPLATES)
template <typename T>
ASIO_STATIC_CONSTEXPR(bool,
is_applicable_property_v = (
is_executor<T>::value
|| conditional<
is_executor<T>::value,
false_type,
is_sender<T>
>::type::value
|| conditional<
is_executor<T>::value,
false_type,
is_scheduler<T>
>::type::value));
#endif 

ASIO_STATIC_CONSTEXPR(bool, is_requirable = false);
ASIO_STATIC_CONSTEXPR(bool, is_preferable = false);
typedef outstanding_work_t polymorphic_query_result_type;

typedef detail::outstanding_work::untracked_t<I> untracked_t;
typedef detail::outstanding_work::tracked_t<I> tracked_t;

ASIO_CONSTEXPR outstanding_work_t()
: value_(-1)
{
}

ASIO_CONSTEXPR outstanding_work_t(untracked_t)
: value_(0)
{
}

ASIO_CONSTEXPR outstanding_work_t(tracked_t)
: value_(1)
{
}

template <typename T>
struct proxy
{
#if defined(ASIO_HAS_DEDUCED_QUERY_MEMBER_TRAIT)
struct type
{
template <typename P>
auto query(ASIO_MOVE_ARG(P) p) const
noexcept(
noexcept(
declval<typename conditional<true, T, P>::type>().query(
ASIO_MOVE_CAST(P)(p))
)
)
-> decltype(
declval<typename conditional<true, T, P>::type>().query(
ASIO_MOVE_CAST(P)(p))
);
};
#else 
typedef T type;
#endif 
};

template <typename T>
struct static_proxy
{
#if defined(ASIO_HAS_DEDUCED_QUERY_STATIC_CONSTEXPR_MEMBER_TRAIT)
struct type
{
template <typename P>
static constexpr auto query(ASIO_MOVE_ARG(P) p)
noexcept(
noexcept(
conditional<true, T, P>::type::query(ASIO_MOVE_CAST(P)(p))
)
)
-> decltype(
conditional<true, T, P>::type::query(ASIO_MOVE_CAST(P)(p))
)
{
return T::query(ASIO_MOVE_CAST(P)(p));
}
};
#else 
typedef T type;
#endif 
};

template <typename T>
struct query_member :
traits::query_member<typename proxy<T>::type, outstanding_work_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename static_proxy<T>::type, outstanding_work_t> {};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename T>
static ASIO_CONSTEXPR
typename query_static_constexpr_member<T>::result_type
static_query()
ASIO_NOEXCEPT_IF((
query_static_constexpr_member<T>::is_noexcept))
{
return query_static_constexpr_member<T>::value();
}

template <typename T>
static ASIO_CONSTEXPR
typename traits::static_query<T, untracked_t>::result_type
static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
traits::static_query<T, untracked_t>::is_valid
>::type* = 0) ASIO_NOEXCEPT
{
return traits::static_query<T, untracked_t>::value();
}

template <typename T>
static ASIO_CONSTEXPR
typename traits::static_query<T, tracked_t>::result_type
static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
!traits::static_query<T, untracked_t>::is_valid
>::type* = 0,
typename enable_if<
traits::static_query<T, tracked_t>::is_valid
>::type* = 0) ASIO_NOEXCEPT
{
return traits::static_query<T, tracked_t>::value();
}

template <typename E,
typename T = decltype(outstanding_work_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= outstanding_work_t::static_query<E>();
#endif 

friend ASIO_CONSTEXPR bool operator==(
const outstanding_work_t& a, const outstanding_work_t& b)
{
return a.value_ == b.value_;
}

friend ASIO_CONSTEXPR bool operator!=(
const outstanding_work_t& a, const outstanding_work_t& b)
{
return a.value_ != b.value_;
}

struct convertible_from_outstanding_work_t
{
ASIO_CONSTEXPR convertible_from_outstanding_work_t(outstanding_work_t)
{
}
};

template <typename Executor>
friend ASIO_CONSTEXPR outstanding_work_t query(
const Executor& ex, convertible_from_outstanding_work_t,
typename enable_if<
can_query<const Executor&, untracked_t>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(ASIO_MSVC) 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&,
outstanding_work_t<>::untracked_t>::value))
#else 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, untracked_t>::value))
#endif 
#endif 
{
return asio::query(ex, untracked_t());
}

template <typename Executor>
friend ASIO_CONSTEXPR outstanding_work_t query(
const Executor& ex, convertible_from_outstanding_work_t,
typename enable_if<
!can_query<const Executor&, untracked_t>::value
>::type* = 0,
typename enable_if<
can_query<const Executor&, tracked_t>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(ASIO_MSVC) 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&,
outstanding_work_t<>::tracked_t>::value))
#else 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, tracked_t>::value))
#endif 
#endif 
{
return asio::query(ex, tracked_t());
}

ASIO_STATIC_CONSTEXPR_DEFAULT_INIT(untracked_t, untracked);
ASIO_STATIC_CONSTEXPR_DEFAULT_INIT(tracked_t, tracked);

#if !defined(ASIO_HAS_CONSTEXPR)
static const outstanding_work_t instance;
#endif 

private:
int value_;
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T outstanding_work_t<I>::static_query_v;
#endif 

#if !defined(ASIO_HAS_CONSTEXPR)
template <int I>
const outstanding_work_t<I> outstanding_work_t<I>::instance;
#endif

template <int I>
const typename outstanding_work_t<I>::untracked_t
outstanding_work_t<I>::untracked;

template <int I>
const typename outstanding_work_t<I>::tracked_t
outstanding_work_t<I>::tracked;

namespace outstanding_work {

template <int I = 0>
struct untracked_t
{
#if defined(ASIO_HAS_VARIABLE_TEMPLATES)
template <typename T>
ASIO_STATIC_CONSTEXPR(bool,
is_applicable_property_v = (
is_executor<T>::value
|| conditional<
is_executor<T>::value,
false_type,
is_sender<T>
>::type::value
|| conditional<
is_executor<T>::value,
false_type,
is_scheduler<T>
>::type::value));
#endif 

ASIO_STATIC_CONSTEXPR(bool, is_requirable = true);
ASIO_STATIC_CONSTEXPR(bool, is_preferable = true);
typedef outstanding_work_t<I> polymorphic_query_result_type;

ASIO_CONSTEXPR untracked_t()
{
}

template <typename T>
struct query_member :
traits::query_member<
typename outstanding_work_t<I>::template proxy<T>::type, untracked_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename outstanding_work_t<I>::template static_proxy<T>::type,
untracked_t> {};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename T>
static ASIO_CONSTEXPR
typename query_static_constexpr_member<T>::result_type
static_query()
ASIO_NOEXCEPT_IF((
query_static_constexpr_member<T>::is_noexcept))
{
return query_static_constexpr_member<T>::value();
}

template <typename T>
static ASIO_CONSTEXPR untracked_t static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
!traits::query_free<T, untracked_t>::is_valid
>::type* = 0,
typename enable_if<
!can_query<T, tracked_t<I> >::value
>::type* = 0) ASIO_NOEXCEPT
{
return untracked_t();
}

template <typename E, typename T = decltype(untracked_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= untracked_t::static_query<E>();
#endif 

static ASIO_CONSTEXPR outstanding_work_t<I> value()
{
return untracked_t();
}

friend ASIO_CONSTEXPR bool operator==(
const untracked_t&, const untracked_t&)
{
return true;
}

friend ASIO_CONSTEXPR bool operator!=(
const untracked_t&, const untracked_t&)
{
return false;
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T untracked_t<I>::static_query_v;
#endif 

template <int I = 0>
struct tracked_t
{
#if defined(ASIO_HAS_VARIABLE_TEMPLATES)
template <typename T>
ASIO_STATIC_CONSTEXPR(bool,
is_applicable_property_v = (
is_executor<T>::value
|| conditional<
is_executor<T>::value,
false_type,
is_sender<T>
>::type::value
|| conditional<
is_executor<T>::value,
false_type,
is_scheduler<T>
>::type::value));
#endif 

ASIO_STATIC_CONSTEXPR(bool, is_requirable = true);
ASIO_STATIC_CONSTEXPR(bool, is_preferable = true);
typedef outstanding_work_t<I> polymorphic_query_result_type;

ASIO_CONSTEXPR tracked_t()
{
}

template <typename T>
struct query_member :
traits::query_member<
typename outstanding_work_t<I>::template proxy<T>::type, tracked_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename outstanding_work_t<I>::template static_proxy<T>::type,
tracked_t> {};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename T>
static ASIO_CONSTEXPR
typename query_static_constexpr_member<T>::result_type
static_query()
ASIO_NOEXCEPT_IF((
query_static_constexpr_member<T>::is_noexcept))
{
return query_static_constexpr_member<T>::value();
}

template <typename E, typename T = decltype(tracked_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= tracked_t::static_query<E>();
#endif 

static ASIO_CONSTEXPR outstanding_work_t<I> value()
{
return tracked_t();
}

friend ASIO_CONSTEXPR bool operator==(
const tracked_t&, const tracked_t&)
{
return true;
}

friend ASIO_CONSTEXPR bool operator!=(
const tracked_t&, const tracked_t&)
{
return false;
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T tracked_t<I>::static_query_v;
#endif 

} 
} 

typedef detail::outstanding_work_t<> outstanding_work_t;

#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr outstanding_work_t outstanding_work;
#else 
namespace { static const outstanding_work_t&
outstanding_work = outstanding_work_t::instance; }
#endif

} 

#if !defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
struct is_applicable_property<T, execution::outstanding_work_t>
: integral_constant<bool,
execution::is_executor<T>::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_sender<T>
>::type::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_scheduler<T>
>::type::value>
{
};

template <typename T>
struct is_applicable_property<T, execution::outstanding_work_t::untracked_t>
: integral_constant<bool,
execution::is_executor<T>::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_sender<T>
>::type::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_scheduler<T>
>::type::value>
{
};

template <typename T>
struct is_applicable_property<T, execution::outstanding_work_t::tracked_t>
: integral_constant<bool,
execution::is_executor<T>::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_sender<T>
>::type::value
|| conditional<
execution::is_executor<T>::value,
false_type,
execution::is_scheduler<T>
>::type::value>
{
};

#endif 

namespace traits {

#if !defined(ASIO_HAS_DEDUCED_QUERY_FREE_TRAIT)

template <typename T>
struct query_free_default<T, execution::outstanding_work_t,
typename enable_if<
can_query<T, execution::outstanding_work_t::untracked_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<T, execution::outstanding_work_t::untracked_t>::value));

typedef execution::outstanding_work_t result_type;
};

template <typename T>
struct query_free_default<T, execution::outstanding_work_t,
typename enable_if<
!can_query<T, execution::outstanding_work_t::untracked_t>::value
&& can_query<T, execution::outstanding_work_t::tracked_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<T, execution::outstanding_work_t::tracked_t>::value));

typedef execution::outstanding_work_t result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
|| !defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)

template <typename T>
struct static_query<T, execution::outstanding_work_t,
typename enable_if<
execution::detail::outstanding_work_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::outstanding_work_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::outstanding_work_t<0>::
query_static_constexpr_member<T>::value();
}
};

template <typename T>
struct static_query<T, execution::outstanding_work_t,
typename enable_if<
!execution::detail::outstanding_work_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::outstanding_work_t<0>::
query_member<T>::is_valid
&& traits::static_query<T,
execution::outstanding_work_t::untracked_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename traits::static_query<T,
execution::outstanding_work_t::untracked_t>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return traits::static_query<T,
execution::outstanding_work_t::untracked_t>::value();
}
};

template <typename T>
struct static_query<T, execution::outstanding_work_t,
typename enable_if<
!execution::detail::outstanding_work_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::outstanding_work_t<0>::
query_member<T>::is_valid
&& !traits::static_query<T,
execution::outstanding_work_t::untracked_t>::is_valid
&& traits::static_query<T,
execution::outstanding_work_t::tracked_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename traits::static_query<T,
execution::outstanding_work_t::tracked_t>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return traits::static_query<T,
execution::outstanding_work_t::tracked_t>::value();
}
};

template <typename T>
struct static_query<T, execution::outstanding_work_t::untracked_t,
typename enable_if<
execution::detail::outstanding_work::untracked_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::outstanding_work::untracked_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::outstanding_work::untracked_t<0>::
query_static_constexpr_member<T>::value();
}
};

template <typename T>
struct static_query<T, execution::outstanding_work_t::untracked_t,
typename enable_if<
!execution::detail::outstanding_work::untracked_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::outstanding_work::untracked_t<0>::
query_member<T>::is_valid
&& !traits::query_free<T,
execution::outstanding_work_t::untracked_t>::is_valid
&& !can_query<T, execution::outstanding_work_t::tracked_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef execution::outstanding_work_t::untracked_t result_type;

static ASIO_CONSTEXPR result_type value()
{
return result_type();
}
};

template <typename T>
struct static_query<T, execution::outstanding_work_t::tracked_t,
typename enable_if<
execution::detail::outstanding_work::tracked_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::outstanding_work::tracked_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::outstanding_work::tracked_t<0>::
query_static_constexpr_member<T>::value();
}
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_STATIC_REQUIRE_TRAIT)

template <typename T>
struct static_require<T, execution::outstanding_work_t::untracked_t,
typename enable_if<
static_query<T, execution::outstanding_work_t::untracked_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid =
(is_same<typename static_query<T,
execution::outstanding_work_t::untracked_t>::result_type,
execution::outstanding_work_t::untracked_t>::value));
};

template <typename T>
struct static_require<T, execution::outstanding_work_t::tracked_t,
typename enable_if<
static_query<T, execution::outstanding_work_t::tracked_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid =
(is_same<typename static_query<T,
execution::outstanding_work_t::tracked_t>::result_type,
execution::outstanding_work_t::tracked_t>::value));
};

#endif 

} 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
