
#ifndef ASIO_IMPL_EXPERIMENTAL_PARALLEL_GROUP_HPP
#define ASIO_IMPL_EXPERIMENTAL_PARALLEL_GROUP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <atomic>
#include <memory>
#include <new>
#include <tuple>
#include "asio/associated_cancellation_slot.hpp"
#include "asio/detail/recycling_allocator.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/dispatch.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {
namespace detail {

template <typename T, typename = void>
struct parallel_group_op_result
{
public:
parallel_group_op_result()
: has_value_(false)
{
}

parallel_group_op_result(parallel_group_op_result&& other)
: has_value_(other.has_value_)
{
if (has_value_)
new (&u_.value_) T(std::move(other.get()));
}

~parallel_group_op_result()
{
if (has_value_)
u_.value_.~T();
}

T& get() noexcept
{
return u_.value_;
}

template <typename... Args>
void emplace(Args&&... args)
{
new (&u_.value_) T(std::forward<Args>(args)...);
has_value_ = true;
}

private:
union u
{
u() {}
~u() {}
char c_;
T value_;
} u_;
bool has_value_;
};

template <typename Handler, typename... Ops>
struct parallel_group_completion_handler
{
typedef typename decay<
typename prefer_result<
typename associated_executor<Handler>::type,
execution::outstanding_work_t::tracked_t
>::type
>::type executor_type;

parallel_group_completion_handler(Handler&& h)
: handler_(std::move(h)),
executor_(
asio::prefer(
asio::get_associated_executor(handler_),
execution::outstanding_work.tracked))
{
}

executor_type get_executor() const noexcept
{
return executor_;
}

void operator()()
{
this->invoke(std::make_index_sequence<sizeof...(Ops)>());
}

template <std::size_t... I>
void invoke(std::index_sequence<I...>)
{
this->invoke(std::tuple_cat(std::move(std::get<I>(args_).get())...));
}

template <typename... Args>
void invoke(std::tuple<Args...>&& args)
{
this->invoke(std::move(args), std::make_index_sequence<sizeof...(Args)>());
}

template <typename... Args, std::size_t... I>
void invoke(std::tuple<Args...>&& args, std::index_sequence<I...>)
{
std::move(handler_)(completion_order_, std::move(std::get<I>(args))...);
}

Handler handler_;
executor_type executor_;
std::array<std::size_t, sizeof...(Ops)> completion_order_{};
std::tuple<
parallel_group_op_result<
typename parallel_op_signature_as_tuple<
typename parallel_op_signature<Ops>::type
>::type
>...
> args_{};
};

template <typename Condition, typename Handler, typename... Ops>
struct parallel_group_state
{
parallel_group_state(Condition&& c, Handler&& h)
: cancellation_condition_(std::move(c)),
handler_(std::move(h))
{
}

std::atomic<unsigned int> completed_{0};

std::atomic<cancellation_type_t> cancel_type_{cancellation_type::none};

std::atomic<unsigned int> cancellations_requested_{sizeof...(Ops)};

std::atomic<unsigned int> outstanding_{sizeof...(Ops)};

asio::cancellation_signal cancellation_signals_[sizeof...(Ops)];

Condition cancellation_condition_;

parallel_group_completion_handler<Handler, Ops...> handler_;
};

template <std::size_t I, typename Condition, typename Handler, typename... Ops>
struct parallel_group_op_handler
{
typedef asio::cancellation_slot cancellation_slot_type;

parallel_group_op_handler(
std::shared_ptr<parallel_group_state<Condition, Handler, Ops...> > state)
: state_(std::move(state))
{
}

cancellation_slot_type get_cancellation_slot() const noexcept
{
return state_->cancellation_signals_[I].slot();
}

template <typename... Args>
void operator()(Args... args)
{
state_->handler_.completion_order_[state_->completed_++] = I;

cancellation_type_t cancel_type = state_->cancellation_condition_(args...);

std::get<I>(state_->handler_.args_).emplace(std::move(args)...);

if (cancel_type != cancellation_type::none)
{
state_->cancel_type_ = cancel_type;

if (state_->cancellations_requested_++ == 0)
for (std::size_t i = 0; i < sizeof...(Ops); ++i)
if (i != I)
state_->cancellation_signals_[i].emit(cancel_type);
}

if (--state_->outstanding_ == 0)
asio::dispatch(std::move(state_->handler_));
}

std::shared_ptr<parallel_group_state<Condition, Handler, Ops...> > state_;
};

template <typename Executor, std::size_t I,
typename Condition, typename Handler, typename... Ops>
struct parallel_group_op_handler_with_executor :
parallel_group_op_handler<I, Condition, Handler, Ops...>
{
typedef parallel_group_op_handler<I, Condition, Handler, Ops...> base_type;
typedef asio::cancellation_slot cancellation_slot_type;
typedef Executor executor_type;

parallel_group_op_handler_with_executor(
std::shared_ptr<parallel_group_state<Condition, Handler, Ops...> > state,
executor_type ex)
: parallel_group_op_handler<I, Condition, Handler, Ops...>(std::move(state))
{
cancel_proxy_ =
&this->state_->cancellation_signals_[I].slot().template
emplace<cancel_proxy>(this->state_, std::move(ex));
}

cancellation_slot_type get_cancellation_slot() const noexcept
{
return cancel_proxy_->signal_.slot();
}

executor_type get_executor() const noexcept
{
return cancel_proxy_->executor_;
}

struct cancel_proxy
{
cancel_proxy(
std::shared_ptr<parallel_group_state<
Condition, Handler, Ops...> > state,
executor_type ex)
: state_(std::move(state)),
executor_(std::move(ex))
{
}

void operator()(cancellation_type_t type)
{
if (auto state = state_.lock())
{
asio::cancellation_signal* sig = &signal_;
asio::dispatch(executor_,
[state, sig, type]{ sig->emit(type); });
}
}

std::weak_ptr<parallel_group_state<Condition, Handler, Ops...> > state_;
asio::cancellation_signal signal_;
executor_type executor_;
};

cancel_proxy* cancel_proxy_;
};

template <std::size_t I, typename Op, typename = void>
struct parallel_group_op_launcher
{
template <typename Condition, typename Handler, typename... Ops>
static void launch(Op& op,
const std::shared_ptr<parallel_group_state<
Condition, Handler, Ops...> >& state)
{
typedef typename associated_executor<Op>::type ex_type;
ex_type ex = asio::get_associated_executor(op);
std::move(op)(
parallel_group_op_handler_with_executor<ex_type, I,
Condition, Handler, Ops...>(state, std::move(ex)));
}
};

template <std::size_t I, typename Op>
struct parallel_group_op_launcher<I, Op,
typename enable_if<
is_same<
typename associated_executor<
Op>::asio_associated_executor_is_unspecialised,
void
>::value
>::type>
{
template <typename Condition, typename Handler, typename... Ops>
static void launch(Op& op,
const std::shared_ptr<parallel_group_state<
Condition, Handler, Ops...> >& state)
{
std::move(op)(
parallel_group_op_handler<I, Condition, Handler, Ops...>(state));
}
};

template <typename Condition, typename Handler, typename... Ops>
struct parallel_group_cancellation_handler
{
parallel_group_cancellation_handler(
std::shared_ptr<parallel_group_state<Condition, Handler, Ops...> > state)
: state_(std::move(state))
{
}

void operator()(cancellation_type_t cancel_type)
{
if (cancel_type != cancellation_type::none)
if (state_->cancellations_requested_++ == 0)
for (std::size_t i = 0; i < sizeof...(Ops); ++i)
state_->cancellation_signals_[i].emit(cancel_type);
}

std::shared_ptr<parallel_group_state<Condition, Handler, Ops...> > state_;
};

template <typename Condition, typename Handler,
typename... Ops, std::size_t... I>
void parallel_group_launch(Condition cancellation_condition, Handler handler,
std::tuple<Ops...>& ops, std::index_sequence<I...>)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef parallel_group_state<Condition, Handler, Ops...> state_type;
std::shared_ptr<state_type> state = std::allocate_shared<state_type>(
asio::detail::recycling_allocator<state_type,
asio::detail::thread_info_base::parallel_group_tag>(),
std::move(cancellation_condition), std::move(handler));

int fold[] = { 0,
( parallel_group_op_launcher<I, Ops>::launch(std::get<I>(ops), state),
0 )...
};
(void)fold;

if ((state->cancellations_requested_ -= sizeof...(Ops)) > 0)
for (auto& signal : state->cancellation_signals_)
signal.emit(state->cancel_type_);

if (slot.is_connected())
slot.template emplace<
parallel_group_cancellation_handler<
Condition, Handler, Ops...> >(state);
}

} 
} 

template <typename R, typename... Args>
class async_result<
experimental::detail::parallel_op_signature_probe,
R(Args...)>
{
public:
typedef experimental::detail::parallel_op_signature_probe_result<
void(Args...)> return_type;

template <typename Initiation, typename... InitArgs>
static return_type initiate(Initiation&&,
experimental::detail::parallel_op_signature_probe, InitArgs&&...)
{
return return_type{};
}
};

template <template <typename, typename> class Associator,
typename Handler, typename... Ops, typename DefaultCandidate>
struct associator<Associator,
experimental::detail::parallel_group_completion_handler<Handler, Ops...>,
DefaultCandidate>
: Associator<Handler, DefaultCandidate>
{
static typename Associator<Handler, DefaultCandidate>::type get(
const experimental::detail::parallel_group_completion_handler<
Handler, Ops...>& h,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<Handler, DefaultCandidate>::get(h.handler_, c);
}
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
