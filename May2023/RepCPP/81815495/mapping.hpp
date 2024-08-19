
#ifndef ASIO_EXECUTION_MAPPING_HPP
#define ASIO_EXECUTION_MAPPING_HPP

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

struct mapping_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = false;

static constexpr bool is_preferable = false;

typedef mapping_t polymorphic_query_result_type;

struct thread_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = true;

static constexpr bool is_preferable = true;

typedef mapping_t polymorphic_query_result_type;

constexpr thread_t();


static constexpr mapping_t value();
};

struct new_thread_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = true;

static constexpr bool is_preferable = true;

typedef mapping_t polymorphic_query_result_type;

constexpr new_thread_t();


static constexpr mapping_t value();
};

struct other_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = true;

static constexpr bool is_preferable = true;

typedef mapping_t polymorphic_query_result_type;

constexpr other_t();


static constexpr mapping_t value();
};

static constexpr thread_t thread;

static constexpr new_thread_t new_thread;

static constexpr other_t other;

constexpr mapping_t();

constexpr mapping_t(thread_t);

constexpr mapping_t(new_thread_t);

constexpr mapping_t(other_t);

friend constexpr bool operator==(
const mapping_t& a, const mapping_t& b) noexcept;

friend constexpr bool operator!=(
const mapping_t& a, const mapping_t& b) noexcept;
};

constexpr mapping_t mapping;

} 

#else 

namespace execution {
namespace detail {
namespace mapping {

template <int I> struct thread_t;
template <int I> struct new_thread_t;
template <int I> struct other_t;

} 

template <int I = 0>
struct mapping_t
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
typedef mapping_t polymorphic_query_result_type;

typedef detail::mapping::thread_t<I> thread_t;
typedef detail::mapping::new_thread_t<I> new_thread_t;
typedef detail::mapping::other_t<I> other_t;

ASIO_CONSTEXPR mapping_t()
: value_(-1)
{
}

ASIO_CONSTEXPR mapping_t(thread_t)
: value_(0)
{
}

ASIO_CONSTEXPR mapping_t(new_thread_t)
: value_(1)
{
}

ASIO_CONSTEXPR mapping_t(other_t)
: value_(2)
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
traits::query_member<typename proxy<T>::type, mapping_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename static_proxy<T>::type, mapping_t> {};

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
typename traits::static_query<T, thread_t>::result_type
static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
traits::static_query<T, thread_t>::is_valid
>::type* = 0) ASIO_NOEXCEPT
{
return traits::static_query<T, thread_t>::value();
}

template <typename T>
static ASIO_CONSTEXPR
typename traits::static_query<T, new_thread_t>::result_type
static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
!traits::static_query<T, thread_t>::is_valid
>::type* = 0,
typename enable_if<
traits::static_query<T, new_thread_t>::is_valid
>::type* = 0) ASIO_NOEXCEPT
{
return traits::static_query<T, new_thread_t>::value();
}

template <typename T>
static ASIO_CONSTEXPR
typename traits::static_query<T, other_t>::result_type
static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
!traits::static_query<T, thread_t>::is_valid
>::type* = 0,
typename enable_if<
!traits::static_query<T, new_thread_t>::is_valid
>::type* = 0,
typename enable_if<
traits::static_query<T, other_t>::is_valid
>::type* = 0) ASIO_NOEXCEPT
{
return traits::static_query<T, other_t>::value();
}

template <typename E, typename T = decltype(mapping_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= mapping_t::static_query<E>();
#endif 

friend ASIO_CONSTEXPR bool operator==(
const mapping_t& a, const mapping_t& b)
{
return a.value_ == b.value_;
}

friend ASIO_CONSTEXPR bool operator!=(
const mapping_t& a, const mapping_t& b)
{
return a.value_ != b.value_;
}

struct convertible_from_mapping_t
{
ASIO_CONSTEXPR convertible_from_mapping_t(mapping_t) {}
};

template <typename Executor>
friend ASIO_CONSTEXPR mapping_t query(
const Executor& ex, convertible_from_mapping_t,
typename enable_if<
can_query<const Executor&, thread_t>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(ASIO_MSVC) 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, mapping_t<>::thread_t>::value))
#else 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, thread_t>::value))
#endif 
#endif 
{
return asio::query(ex, thread_t());
}

template <typename Executor>
friend ASIO_CONSTEXPR mapping_t query(
const Executor& ex, convertible_from_mapping_t,
typename enable_if<
!can_query<const Executor&, thread_t>::value
>::type* = 0,
typename enable_if<
can_query<const Executor&, new_thread_t>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(ASIO_MSVC) 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, mapping_t<>::new_thread_t>::value))
#else 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, new_thread_t>::value))
#endif 
#endif 
{
return asio::query(ex, new_thread_t());
}

template <typename Executor>
friend ASIO_CONSTEXPR mapping_t query(
const Executor& ex, convertible_from_mapping_t,
typename enable_if<
!can_query<const Executor&, thread_t>::value
>::type* = 0,
typename enable_if<
!can_query<const Executor&, new_thread_t>::value
>::type* = 0,
typename enable_if<
can_query<const Executor&, other_t>::value
>::type* = 0)
#if !defined(__clang__) 
#if defined(ASIO_MSVC) 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, mapping_t<>::other_t>::value))
#else 
ASIO_NOEXCEPT_IF((
is_nothrow_query<const Executor&, other_t>::value))
#endif 
#endif 
{
return asio::query(ex, other_t());
}

ASIO_STATIC_CONSTEXPR_DEFAULT_INIT(thread_t, thread);
ASIO_STATIC_CONSTEXPR_DEFAULT_INIT(new_thread_t, new_thread);
ASIO_STATIC_CONSTEXPR_DEFAULT_INIT(other_t, other);

#if !defined(ASIO_HAS_CONSTEXPR)
static const mapping_t instance;
#endif 

private:
int value_;
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T mapping_t<I>::static_query_v;
#endif 

#if !defined(ASIO_HAS_CONSTEXPR)
template <int I>
const mapping_t<I> mapping_t<I>::instance;
#endif

template <int I>
const typename mapping_t<I>::thread_t mapping_t<I>::thread;

template <int I>
const typename mapping_t<I>::new_thread_t mapping_t<I>::new_thread;

template <int I>
const typename mapping_t<I>::other_t mapping_t<I>::other;

namespace mapping {

template <int I = 0>
struct thread_t
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
typedef mapping_t<I> polymorphic_query_result_type;

ASIO_CONSTEXPR thread_t()
{
}

template <typename T>
struct query_member :
traits::query_member<
typename mapping_t<I>::template proxy<T>::type, thread_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename mapping_t<I>::template static_proxy<T>::type, thread_t> {};

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
static ASIO_CONSTEXPR thread_t static_query(
typename enable_if<
!query_static_constexpr_member<T>::is_valid
>::type* = 0,
typename enable_if<
!query_member<T>::is_valid
>::type* = 0,
typename enable_if<
!traits::query_free<T, thread_t>::is_valid
>::type* = 0,
typename enable_if<
!can_query<T, new_thread_t<I> >::value
>::type* = 0,
typename enable_if<
!can_query<T, other_t<I> >::value
>::type* = 0) ASIO_NOEXCEPT
{
return thread_t();
}

template <typename E, typename T = decltype(thread_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= thread_t::static_query<E>();
#endif 

static ASIO_CONSTEXPR mapping_t<I> value()
{
return thread_t();
}

friend ASIO_CONSTEXPR bool operator==(
const thread_t&, const thread_t&)
{
return true;
}

friend ASIO_CONSTEXPR bool operator!=(
const thread_t&, const thread_t&)
{
return false;
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T thread_t<I>::static_query_v;
#endif 

template <int I = 0>
struct new_thread_t
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
typedef mapping_t<I> polymorphic_query_result_type;

ASIO_CONSTEXPR new_thread_t()
{
}

template <typename T>
struct query_member :
traits::query_member<
typename mapping_t<I>::template proxy<T>::type, new_thread_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename mapping_t<I>::template static_proxy<T>::type, new_thread_t> {};

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

template <typename E, typename T = decltype(new_thread_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= new_thread_t::static_query<E>();
#endif 

static ASIO_CONSTEXPR mapping_t<I> value()
{
return new_thread_t();
}

friend ASIO_CONSTEXPR bool operator==(
const new_thread_t&, const new_thread_t&)
{
return true;
}

friend ASIO_CONSTEXPR bool operator!=(
const new_thread_t&, const new_thread_t&)
{
return false;
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T new_thread_t<I>::static_query_v;
#endif 

template <int I>
struct other_t
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
typedef mapping_t<I> polymorphic_query_result_type;

ASIO_CONSTEXPR other_t()
{
}

template <typename T>
struct query_member :
traits::query_member<
typename mapping_t<I>::template proxy<T>::type, other_t> {};

template <typename T>
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename mapping_t<I>::template static_proxy<T>::type, other_t> {};

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

template <typename E, typename T = decltype(other_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= other_t::static_query<E>();
#endif 

static ASIO_CONSTEXPR mapping_t<I> value()
{
return other_t();
}

friend ASIO_CONSTEXPR bool operator==(
const other_t&, const other_t&)
{
return true;
}

friend ASIO_CONSTEXPR bool operator!=(
const other_t&, const other_t&)
{
return false;
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <int I> template <typename E, typename T>
const T other_t<I>::static_query_v;
#endif 

} 
} 

typedef detail::mapping_t<> mapping_t;

#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr mapping_t mapping;
#else 
namespace { static const mapping_t& mapping = mapping_t::instance; }
#endif

} 

#if !defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T>
struct is_applicable_property<T, execution::mapping_t>
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
struct is_applicable_property<T, execution::mapping_t::thread_t>
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
struct is_applicable_property<T, execution::mapping_t::new_thread_t>
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
struct is_applicable_property<T, execution::mapping_t::other_t>
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
struct query_free_default<T, execution::mapping_t,
typename enable_if<
can_query<T, execution::mapping_t::thread_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<T, execution::mapping_t::thread_t>::value));

typedef execution::mapping_t result_type;
};

template <typename T>
struct query_free_default<T, execution::mapping_t,
typename enable_if<
!can_query<T, execution::mapping_t::thread_t>::value
&& can_query<T, execution::mapping_t::new_thread_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<T, execution::mapping_t::new_thread_t>::value));

typedef execution::mapping_t result_type;
};

template <typename T>
struct query_free_default<T, execution::mapping_t,
typename enable_if<
!can_query<T, execution::mapping_t::thread_t>::value
&& !can_query<T, execution::mapping_t::new_thread_t>::value
&& can_query<T, execution::mapping_t::other_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept =
(is_nothrow_query<T, execution::mapping_t::other_t>::value));

typedef execution::mapping_t result_type;
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
|| !defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)

template <typename T>
struct static_query<T, execution::mapping_t,
typename enable_if<
execution::detail::mapping_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::mapping_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::mapping_t<0>::
query_static_constexpr_member<T>::value();
}
};

template <typename T>
struct static_query<T, execution::mapping_t,
typename enable_if<
!execution::detail::mapping_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::mapping_t<0>::
query_member<T>::is_valid
&& traits::static_query<T, execution::mapping_t::thread_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename traits::static_query<T,
execution::mapping_t::thread_t>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return traits::static_query<T, execution::mapping_t::thread_t>::value();
}
};

template <typename T>
struct static_query<T, execution::mapping_t,
typename enable_if<
!execution::detail::mapping_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::mapping_t<0>::
query_member<T>::is_valid
&& !traits::static_query<T, execution::mapping_t::thread_t>::is_valid
&& traits::static_query<T, execution::mapping_t::new_thread_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename traits::static_query<T,
execution::mapping_t::new_thread_t>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return traits::static_query<T, execution::mapping_t::new_thread_t>::value();
}
};

template <typename T>
struct static_query<T, execution::mapping_t,
typename enable_if<
!execution::detail::mapping_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::mapping_t<0>::
query_member<T>::is_valid
&& !traits::static_query<T, execution::mapping_t::thread_t>::is_valid
&& !traits::static_query<T, execution::mapping_t::new_thread_t>::is_valid
&& traits::static_query<T, execution::mapping_t::other_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename traits::static_query<T,
execution::mapping_t::other_t>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return traits::static_query<T, execution::mapping_t::other_t>::value();
}
};

template <typename T>
struct static_query<T, execution::mapping_t::thread_t,
typename enable_if<
execution::detail::mapping::thread_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::mapping::thread_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::mapping::thread_t<0>::
query_static_constexpr_member<T>::value();
}
};

template <typename T>
struct static_query<T, execution::mapping_t::thread_t,
typename enable_if<
!execution::detail::mapping::thread_t<0>::
query_static_constexpr_member<T>::is_valid
&& !execution::detail::mapping::thread_t<0>::
query_member<T>::is_valid
&& !traits::query_free<T, execution::mapping_t::thread_t>::is_valid
&& !can_query<T, execution::mapping_t::new_thread_t>::value
&& !can_query<T, execution::mapping_t::other_t>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef execution::mapping_t::thread_t result_type;

static ASIO_CONSTEXPR result_type value()
{
return result_type();
}
};

template <typename T>
struct static_query<T, execution::mapping_t::new_thread_t,
typename enable_if<
execution::detail::mapping::new_thread_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::mapping::new_thread_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::mapping::new_thread_t<0>::
query_static_constexpr_member<T>::value();
}
};

template <typename T>
struct static_query<T, execution::mapping_t::other_t,
typename enable_if<
execution::detail::mapping::other_t<0>::
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::detail::mapping::other_t<0>::
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::detail::mapping::other_t<0>::
query_static_constexpr_member<T>::value();
}
};

#endif 

#if !defined(ASIO_HAS_DEDUCED_STATIC_REQUIRE_TRAIT)

template <typename T>
struct static_require<T, execution::mapping_t::thread_t,
typename enable_if<
static_query<T, execution::mapping_t::thread_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid =
(is_same<typename static_query<T,
execution::mapping_t::thread_t>::result_type,
execution::mapping_t::thread_t>::value));
};

template <typename T>
struct static_require<T, execution::mapping_t::new_thread_t,
typename enable_if<
static_query<T, execution::mapping_t::new_thread_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid =
(is_same<typename static_query<T,
execution::mapping_t::new_thread_t>::result_type,
execution::mapping_t::new_thread_t>::value));
};

template <typename T>
struct static_require<T, execution::mapping_t::other_t,
typename enable_if<
static_query<T, execution::mapping_t::other_t>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid =
(is_same<typename static_query<T,
execution::mapping_t::other_t>::result_type,
execution::mapping_t::other_t>::value));
};

#endif 

} 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
