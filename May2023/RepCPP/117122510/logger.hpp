

#ifndef GKO_PUBLIC_CORE_LOG_LOGGER_HPP_
#define GKO_PUBLIC_CORE_LOG_LOGGER_HPP_


#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils_helper.hpp>


namespace gko {



template <typename ValueType>
class array;
class Executor;
class LinOp;
class LinOpFactory;
class PolymorphicObject;
class Operation;
class stopping_status;


namespace stop {
class Criterion;
}  


namespace log {



class Logger {
public:

using mask_type = gko::uint64;


static constexpr size_type event_count_max = sizeof(mask_type) * byte_size;


static constexpr mask_type all_events_mask = ~mask_type{0};


#define GKO_LOGGER_REGISTER_EVENT(_id, _event_name, ...)             \
protected:                                                           \
virtual void on_##_event_name(__VA_ARGS__) const {}              \
\
public:                                                              \
template <size_type Event, typename... Params>                   \
std::enable_if_t<Event == _id && (_id < event_count_max)> on(    \
Params&&... params) const                                    \
{                                                                \
if (enabled_events_ & (mask_type{1} << _id)) {               \
this->on_##_event_name(std::forward<Params>(params)...); \
}                                                            \
}                                                                \
static constexpr size_type _event_name{_id};                     \
static constexpr mask_type _event_name##_mask{mask_type{1} << _id};


GKO_LOGGER_REGISTER_EVENT(0, allocation_started, const Executor* exec,
const size_type& num_bytes)


GKO_LOGGER_REGISTER_EVENT(1, allocation_completed, const Executor* exec,
const size_type& num_bytes,
const uintptr& location)


GKO_LOGGER_REGISTER_EVENT(2, free_started, const Executor* exec,
const uintptr& location)


GKO_LOGGER_REGISTER_EVENT(3, free_completed, const Executor* exec,
const uintptr& location)


GKO_LOGGER_REGISTER_EVENT(4, copy_started, const Executor* exec_from,
const Executor* exec_to, const uintptr& loc_from,
const uintptr& loc_to, const size_type& num_bytes)


GKO_LOGGER_REGISTER_EVENT(5, copy_completed, const Executor* exec_from,
const Executor* exec_to, const uintptr& loc_from,
const uintptr& loc_to, const size_type& num_bytes)


GKO_LOGGER_REGISTER_EVENT(6, operation_launched, const Executor* exec,
const Operation* op)


GKO_LOGGER_REGISTER_EVENT(7, operation_completed, const Executor* exec,
const Operation* op)


GKO_LOGGER_REGISTER_EVENT(8, polymorphic_object_create_started,
const Executor* exec, const PolymorphicObject* po)


GKO_LOGGER_REGISTER_EVENT(9, polymorphic_object_create_completed,
const Executor* exec,
const PolymorphicObject* input,
const PolymorphicObject* output)


GKO_LOGGER_REGISTER_EVENT(10, polymorphic_object_copy_started,
const Executor* exec,
const PolymorphicObject* input,
const PolymorphicObject* output)


GKO_LOGGER_REGISTER_EVENT(11, polymorphic_object_copy_completed,
const Executor* exec,
const PolymorphicObject* input,
const PolymorphicObject* output)


GKO_LOGGER_REGISTER_EVENT(12, polymorphic_object_deleted,
const Executor* exec, const PolymorphicObject* po)


GKO_LOGGER_REGISTER_EVENT(13, linop_apply_started, const LinOp* A,
const LinOp* b, const LinOp* x)


GKO_LOGGER_REGISTER_EVENT(14, linop_apply_completed, const LinOp* A,
const LinOp* b, const LinOp* x)


GKO_LOGGER_REGISTER_EVENT(15, linop_advanced_apply_started, const LinOp* A,
const LinOp* alpha, const LinOp* b,
const LinOp* beta, const LinOp* x)


GKO_LOGGER_REGISTER_EVENT(16, linop_advanced_apply_completed,
const LinOp* A, const LinOp* alpha,
const LinOp* b, const LinOp* beta, const LinOp* x)


GKO_LOGGER_REGISTER_EVENT(17, linop_factory_generate_started,
const LinOpFactory* factory, const LinOp* input)


GKO_LOGGER_REGISTER_EVENT(18, linop_factory_generate_completed,
const LinOpFactory* factory, const LinOp* input,
const LinOp* output)


GKO_LOGGER_REGISTER_EVENT(19, criterion_check_started,
const stop::Criterion* criterion,
const size_type& it, const LinOp* r,
const LinOp* tau, const LinOp* x,
const uint8& stopping_id,
const bool& set_finalized)


GKO_LOGGER_REGISTER_EVENT(
20, criterion_check_completed, const stop::Criterion* criterion,
const size_type& it, const LinOp* r, const LinOp* tau, const LinOp* x,
const uint8& stopping_id, const bool& set_finalized,
const array<stopping_status>* status, const bool& one_changed,
const bool& all_converged)
protected:

virtual void on_criterion_check_completed(
const stop::Criterion* criterion, const size_type& it, const LinOp* r,
const LinOp* tau, const LinOp* implicit_tau_sq, const LinOp* x,
const uint8& stopping_id, const bool& set_finalized,
const array<stopping_status>* status, const bool& one_changed,
const bool& all_converged) const
{
this->on_criterion_check_completed(criterion, it, r, tau, x,
stopping_id, set_finalized, status,
one_changed, all_converged);
}

public:
static constexpr size_type iteration_complete{21};
static constexpr mask_type iteration_complete_mask{mask_type{1} << 21};

template <size_type Event, typename... Params>
std::enable_if_t<Event == 21 && (21 < event_count_max)> on(
Params&&... params) const
{
if (enabled_events_ & (mask_type{1} << 21)) {
this->on_iteration_complete(std::forward<Params>(params)...);
}
}

protected:

[[deprecated(
"Please use the version with the additional stopping "
"information.")]] virtual void
on_iteration_complete(const LinOp* solver, const size_type& it,
const LinOp* r, const LinOp* x = nullptr,
const LinOp* tau = nullptr) const
{}


[[deprecated(
"Please use the version with the additional stopping "
"information.")]] virtual void
on_iteration_complete(const LinOp* solver, const size_type& it,
const LinOp* r, const LinOp* x, const LinOp* tau,
const LinOp* implicit_tau_sq) const
{
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5211, 4973, 4974)
#endif
this->on_iteration_complete(solver, it, r, x, tau);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}


virtual void on_iteration_complete(const LinOp* solver, const LinOp* b,
const LinOp* x, const size_type& it,
const LinOp* r, const LinOp* tau,
const LinOp* implicit_tau_sq,
const array<stopping_status>* status,
bool stopped) const
{
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 5211, 4973, 4974)
#endif
this->on_iteration_complete(solver, it, r, x, tau, implicit_tau_sq);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

public:

GKO_LOGGER_REGISTER_EVENT(22, polymorphic_object_move_started,
const Executor* exec,
const PolymorphicObject* input,
const PolymorphicObject* output)


GKO_LOGGER_REGISTER_EVENT(23, polymorphic_object_move_completed,
const Executor* exec,
const PolymorphicObject* input,
const PolymorphicObject* output)

#undef GKO_LOGGER_REGISTER_EVENT


static constexpr mask_type executor_events_mask =
allocation_started_mask | allocation_completed_mask |
free_started_mask | free_completed_mask | copy_started_mask |
copy_completed_mask;


static constexpr mask_type operation_events_mask =
operation_launched_mask | operation_completed_mask;


static constexpr mask_type polymorphic_object_events_mask =
polymorphic_object_create_started_mask |
polymorphic_object_create_completed_mask |
polymorphic_object_copy_started_mask |
polymorphic_object_copy_completed_mask |
polymorphic_object_move_started_mask |
polymorphic_object_move_completed_mask |
polymorphic_object_deleted_mask;


static constexpr mask_type linop_events_mask =
linop_apply_started_mask | linop_apply_completed_mask |
linop_advanced_apply_started_mask | linop_advanced_apply_completed_mask;


static constexpr mask_type linop_factory_events_mask =
linop_factory_generate_started_mask |
linop_factory_generate_completed_mask;


static constexpr mask_type criterion_events_mask =
criterion_check_started_mask | criterion_check_completed_mask;


virtual bool needs_propagation() const { return false; }

virtual ~Logger() = default;

protected:

[[deprecated("use single-parameter constructor")]] explicit Logger(
std::shared_ptr<const gko::Executor> exec,
const mask_type& enabled_events = all_events_mask)
: Logger{enabled_events}
{}


explicit Logger(const mask_type& enabled_events = all_events_mask)
: enabled_events_{enabled_events}
{}

private:
mask_type enabled_events_;
};



class Loggable {
public:
virtual ~Loggable() = default;


virtual void add_logger(std::shared_ptr<const Logger> logger) = 0;


virtual void remove_logger(const Logger* logger) = 0;

void remove_logger(ptr_param<const Logger> logger)
{
remove_logger(logger.get());
}


virtual const std::vector<std::shared_ptr<const Logger>>& get_loggers()
const = 0;


virtual void clear_loggers() = 0;
};



template <typename ConcreteLoggable, typename PolymorphicBase = Loggable>
class EnableLogging : public PolymorphicBase {
public:
void add_logger(std::shared_ptr<const Logger> logger) override
{
loggers_.push_back(logger);
}

void remove_logger(const Logger* logger) override
{
auto idx =
find_if(begin(loggers_), end(loggers_),
[&logger](const auto& l) { return l.get() == logger; });
if (idx != end(loggers_)) {
loggers_.erase(idx);
} else {
throw OutOfBoundsError(__FILE__, __LINE__, loggers_.size(),
loggers_.size());
}
}

void remove_logger(ptr_param<const Logger> logger)
{
remove_logger(logger.get());
}

const std::vector<std::shared_ptr<const Logger>>& get_loggers()
const override
{
return loggers_;
}

void clear_loggers() override { loggers_.clear(); }

private:

template <size_type Event, typename ConcreteLoggableT, typename = void>
struct propagate_log_helper {
template <typename... Args>
static void propagate_log(const ConcreteLoggableT*, Args&&...)
{}
};

template <size_type Event, typename ConcreteLoggableT>
struct propagate_log_helper<
Event, ConcreteLoggableT,
xstd::void_t<
decltype(std::declval<ConcreteLoggableT>().get_executor())>> {
template <typename... Args>
static void propagate_log(const ConcreteLoggableT* loggable,
Args&&... args)
{
const auto exec = loggable->get_executor();
if (exec->should_propagate_log()) {
for (auto& logger : exec->get_loggers()) {
if (logger->needs_propagation()) {
logger->template on<Event>(std::forward<Args>(args)...);
}
}
}
}
};

protected:
template <size_type Event, typename... Params>
void log(Params&&... params) const
{
propagate_log_helper<Event, ConcreteLoggable>::propagate_log(
static_cast<const ConcreteLoggable*>(this),
std::forward<Params>(params)...);
for (auto& logger : loggers_) {
logger->template on<Event>(std::forward<Params>(params)...);
}
}

std::vector<std::shared_ptr<const Logger>> loggers_;
};


}  
}  


#endif  
