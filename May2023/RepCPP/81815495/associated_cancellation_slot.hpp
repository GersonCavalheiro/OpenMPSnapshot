
#ifndef ASIO_ASSOCIATED_CANCELLATION_SLOT_HPP
#define ASIO_ASSOCIATED_CANCELLATION_SLOT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/associator.hpp"
#include "asio/cancellation_signal.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename T, typename CancellationSlot>
struct associated_cancellation_slot;

namespace detail {

template <typename T, typename = void>
struct has_cancellation_slot_type : false_type
{
};

template <typename T>
struct has_cancellation_slot_type<T,
typename void_type<typename T::cancellation_slot_type>::type>
: true_type
{
};

template <typename T, typename S, typename = void, typename = void>
struct associated_cancellation_slot_impl
{
typedef void asio_associated_cancellation_slot_is_unspecialised;

typedef S type;

static type get(const T&, const S& s = S()) ASIO_NOEXCEPT
{
return s;
}
};

template <typename T, typename S>
struct associated_cancellation_slot_impl<T, S,
typename void_type<typename T::cancellation_slot_type>::type>
{
typedef typename T::cancellation_slot_type type;

static type get(const T& t, const S& = S()) ASIO_NOEXCEPT
{
return t.get_cancellation_slot();
}
};

template <typename T, typename S>
struct associated_cancellation_slot_impl<T, S,
typename enable_if<
!has_cancellation_slot_type<T>::value
>::type,
typename void_type<
typename associator<associated_cancellation_slot, T, S>::type
>::type> : associator<associated_cancellation_slot, T, S>
{
};

} 


template <typename T, typename CancellationSlot = cancellation_slot>
struct associated_cancellation_slot
#if !defined(GENERATING_DOCUMENTATION)
: detail::associated_cancellation_slot_impl<T, CancellationSlot>
#endif 
{
#if defined(GENERATING_DOCUMENTATION)
typedef see_below type;

static type get(const T& t,
const CancellationSlot& s = CancellationSlot()) ASIO_NOEXCEPT;
#endif 
};


template <typename T>
inline typename associated_cancellation_slot<T>::type
get_associated_cancellation_slot(const T& t) ASIO_NOEXCEPT
{
return associated_cancellation_slot<T>::get(t);
}


template <typename T, typename CancellationSlot>
inline typename associated_cancellation_slot<T, CancellationSlot>::type
get_associated_cancellation_slot(const T& t,
const CancellationSlot& st) ASIO_NOEXCEPT
{
return associated_cancellation_slot<T, CancellationSlot>::get(t, st);
}

#if defined(ASIO_HAS_ALIAS_TEMPLATES)

template <typename T, typename CancellationSlot = cancellation_slot>
using associated_cancellation_slot_t =
typename associated_cancellation_slot<T, CancellationSlot>::type;

#endif 

namespace detail {

template <typename T, typename S, typename = void>
struct associated_cancellation_slot_forwarding_base
{
};

template <typename T, typename S>
struct associated_cancellation_slot_forwarding_base<T, S,
typename enable_if<
is_same<
typename associated_cancellation_slot<T,
S>::asio_associated_cancellation_slot_is_unspecialised,
void
>::value
>::type>
{
typedef void asio_associated_cancellation_slot_is_unspecialised;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
