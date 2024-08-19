
#ifndef ASIO_CANCELLATION_STATE_HPP
#define ASIO_CANCELLATION_STATE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cassert>
#include <new>
#include <utility>
#include "asio/cancellation_signal.hpp"
#include "asio/detail/cstddef.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <cancellation_type_t Mask>
struct cancellation_filter
{
cancellation_type_t operator()(
cancellation_type_t type) const ASIO_NOEXCEPT
{
return type & Mask;
}
};

typedef cancellation_filter<cancellation_type::none>
disable_cancellation;

typedef cancellation_filter<cancellation_type::terminal>
enable_terminal_cancellation;

#if defined(GENERATING_DOCUMENTATION)

typedef cancellation_filter<
cancellation_type::terminal | cancellation_type::partial>
enable_partial_cancellation;

typedef cancellation_filter<cancellation_type::terminal
| cancellation_type::partial | cancellation_type::total>
enable_total_cancellation;

#else 

typedef cancellation_filter<
static_cast<cancellation_type_t>(
static_cast<unsigned int>(cancellation_type::terminal)
| static_cast<unsigned int>(cancellation_type::partial))>
enable_partial_cancellation;

typedef cancellation_filter<
static_cast<cancellation_type_t>(
static_cast<unsigned int>(cancellation_type::terminal)
| static_cast<unsigned int>(cancellation_type::partial)
| static_cast<unsigned int>(cancellation_type::total))>
enable_total_cancellation;

#endif 

class cancellation_state
{
public:
ASIO_CONSTEXPR cancellation_state() ASIO_NOEXCEPT
: impl_(0)
{
}


template <typename CancellationSlot>
ASIO_CONSTEXPR explicit cancellation_state(CancellationSlot slot)
: impl_(slot.is_connected() ? &slot.template emplace<impl<> >() : 0)
{
}


template <typename CancellationSlot, typename Filter>
ASIO_CONSTEXPR cancellation_state(CancellationSlot slot, Filter filter)
: impl_(slot.is_connected()
? &slot.template emplace<impl<Filter, Filter> >(filter, filter)
: 0)
{
}


template <typename CancellationSlot, typename InFilter, typename OutFilter>
ASIO_CONSTEXPR cancellation_state(CancellationSlot slot,
InFilter in_filter, OutFilter out_filter)
: impl_(slot.is_connected()
? &slot.template emplace<impl<InFilter, OutFilter> >(
ASIO_MOVE_CAST(InFilter)(in_filter),
ASIO_MOVE_CAST(OutFilter)(out_filter))
: 0)
{
}


ASIO_CONSTEXPR cancellation_slot slot() const ASIO_NOEXCEPT
{
return impl_ ? impl_->signal_.slot() : cancellation_slot();
}

cancellation_type_t cancelled() const ASIO_NOEXCEPT
{
return impl_ ? impl_->cancelled_ : cancellation_type_t();
}

void clear(cancellation_type_t mask = cancellation_type::all)
ASIO_NOEXCEPT
{
if (impl_)
impl_->cancelled_ &= ~mask;
}

private:
struct impl_base
{
impl_base()
: cancelled_()
{
}

cancellation_signal signal_;
cancellation_type_t cancelled_;
};

template <
typename InFilter = enable_terminal_cancellation,
typename OutFilter = InFilter>
struct impl : impl_base
{
impl()
: in_filter_(),
out_filter_()
{
}

impl(InFilter in_filter, OutFilter out_filter)
: in_filter_(ASIO_MOVE_CAST(InFilter)(in_filter)),
out_filter_(ASIO_MOVE_CAST(OutFilter)(out_filter))
{
}

void operator()(cancellation_type_t in)
{
this->cancelled_ = in_filter_(in);
cancellation_type_t out = out_filter_(this->cancelled_);
if (out != cancellation_type::none)
this->signal_.emit(out);
}

InFilter in_filter_;
OutFilter out_filter_;
};

impl_base* impl_;
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
