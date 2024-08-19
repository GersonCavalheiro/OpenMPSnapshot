
#ifndef ASIO_EXECUTION_ALLOCATOR_HPP
#define ASIO_EXECUTION_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution/scheduler.hpp"
#include "asio/execution/sender.hpp"
#include "asio/is_applicable_property.hpp"
#include "asio/traits/query_static_constexpr_member.hpp"
#include "asio/traits/static_query.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(GENERATING_DOCUMENTATION)

namespace execution {

template <typename ProtoAllocator>
struct allocator_t
{
template <typename T>
static constexpr bool is_applicable_property_v =
is_executor_v<T> || is_sender_v<T> || is_scheduler_v<T>;

static constexpr bool is_requirable = true;

static constexpr bool is_preferable = true;

constexpr allocator_t();


constexpr ProtoAllocator value() const;


template <typename OtherAllocator>
allocator_t<OtherAllocator operator()(const OtherAllocator& a);
};

constexpr allocator_t<void> allocator;

} 

#else 

namespace execution {

template <typename ProtoAllocator>
struct allocator_t
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
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename static_proxy<T>::type, allocator_t> {};

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

template <typename E, typename T = decltype(allocator_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= allocator_t::static_query<E>();
#endif 

ASIO_CONSTEXPR ProtoAllocator value() const
{
return a_;
}

private:
friend struct allocator_t<void>;

explicit ASIO_CONSTEXPR allocator_t(const ProtoAllocator& a)
: a_(a)
{
}

ProtoAllocator a_;
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename ProtoAllocator> template <typename E, typename T>
const T allocator_t<ProtoAllocator>::static_query_v;
#endif 

template <>
struct allocator_t<void>
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

ASIO_CONSTEXPR allocator_t()
{
}

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
struct query_static_constexpr_member :
traits::query_static_constexpr_member<
typename static_proxy<T>::type, allocator_t> {};

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

template <typename E, typename T = decltype(allocator_t::static_query<E>())>
static ASIO_CONSTEXPR const T static_query_v
= allocator_t::static_query<E>();
#endif 

template <typename OtherProtoAllocator>
ASIO_CONSTEXPR allocator_t<OtherProtoAllocator> operator()(
const OtherProtoAllocator& a) const
{
return allocator_t<OtherProtoAllocator>(a);
}
};

#if defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
&& defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)
template <typename E, typename T>
const T allocator_t<void>::static_query_v;
#endif 

#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr allocator_t<void> allocator;
#else 
template <typename T>
struct allocator_instance
{
static allocator_t<T> instance;
};

template <typename T>
allocator_t<T> allocator_instance<T>::instance;

namespace {
static const allocator_t<void>& allocator = allocator_instance<void>::instance;
} 
#endif

} 

#if !defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename ProtoAllocator>
struct is_applicable_property<T, execution::allocator_t<ProtoAllocator> >
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

#if !defined(ASIO_HAS_DEDUCED_STATIC_QUERY_TRAIT) \
|| !defined(ASIO_HAS_SFINAE_VARIABLE_TEMPLATES)

template <typename T, typename ProtoAllocator>
struct static_query<T, execution::allocator_t<ProtoAllocator>,
typename enable_if<
execution::allocator_t<ProtoAllocator>::template
query_static_constexpr_member<T>::is_valid
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);

typedef typename execution::allocator_t<ProtoAllocator>::template
query_static_constexpr_member<T>::result_type result_type;

static ASIO_CONSTEXPR result_type value()
{
return execution::allocator_t<ProtoAllocator>::template
query_static_constexpr_member<T>::value();
}
};

#endif 

} 

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
