



#ifndef GMOCK_INCLUDE_GMOCK_GMOCK_ACTIONS_H_
#define GMOCK_INCLUDE_GMOCK_GMOCK_ACTIONS_H_

#ifndef _WIN32_WCE
# include <errno.h>
#endif

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "gmock/internal/gmock-internal-utils.h"
#include "gmock/internal/gmock-port.h"
#include "gmock/internal/gmock-pp.h"

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif

namespace testing {


namespace internal {

template <typename T, bool kDefaultConstructible>
struct BuiltInDefaultValueGetter {
static T Get() { return T(); }
};
template <typename T>
struct BuiltInDefaultValueGetter<T, false> {
static T Get() {
Assert(false, __FILE__, __LINE__,
"Default action undefined for the function return type.");
return internal::Invalid<T>();
}
};

template <typename T>
class BuiltInDefaultValue {
public:
static bool Exists() {
return ::std::is_default_constructible<T>::value;
}

static T Get() {
return BuiltInDefaultValueGetter<
T, ::std::is_default_constructible<T>::value>::Get();
}
};

template <typename T>
class BuiltInDefaultValue<const T> {
public:
static bool Exists() { return BuiltInDefaultValue<T>::Exists(); }
static T Get() { return BuiltInDefaultValue<T>::Get(); }
};

template <typename T>
class BuiltInDefaultValue<T*> {
public:
static bool Exists() { return true; }
static T* Get() { return nullptr; }
};

#define GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(type, value) \
template <> \
class BuiltInDefaultValue<type> { \
public: \
static bool Exists() { return true; } \
static type Get() { return value; } \
}

GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(void, );  
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(::std::string, "");
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(bool, false);
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(unsigned char, '\0');
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(signed char, '\0');
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(char, '\0');

#if GMOCK_WCHAR_T_IS_NATIVE_
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(wchar_t, 0U);  
#endif

GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(unsigned short, 0U);  
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(signed short, 0);     
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(unsigned int, 0U);
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(signed int, 0);
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(unsigned long, 0UL);  
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(signed long, 0L);     
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(unsigned long long, 0);  
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(signed long long, 0);  
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(float, 0);
GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_(double, 0);

#undef GMOCK_DEFINE_DEFAULT_ACTION_FOR_RETURN_TYPE_

template <typename P, typename Q>
using disjunction = typename ::std::conditional<P::value, P, Q>::type;

}  

template <typename T>
class DefaultValue {
public:
static void Set(T x) {
delete producer_;
producer_ = new FixedValueProducer(x);
}

typedef T (*FactoryFunction)();
static void SetFactory(FactoryFunction factory) {
delete producer_;
producer_ = new FactoryValueProducer(factory);
}

static void Clear() {
delete producer_;
producer_ = nullptr;
}

static bool IsSet() { return producer_ != nullptr; }

static bool Exists() {
return IsSet() || internal::BuiltInDefaultValue<T>::Exists();
}

static T Get() {
return producer_ == nullptr ? internal::BuiltInDefaultValue<T>::Get()
: producer_->Produce();
}

private:
class ValueProducer {
public:
virtual ~ValueProducer() {}
virtual T Produce() = 0;
};

class FixedValueProducer : public ValueProducer {
public:
explicit FixedValueProducer(T value) : value_(value) {}
T Produce() override { return value_; }

private:
const T value_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(FixedValueProducer);
};

class FactoryValueProducer : public ValueProducer {
public:
explicit FactoryValueProducer(FactoryFunction factory)
: factory_(factory) {}
T Produce() override { return factory_(); }

private:
const FactoryFunction factory_;
GTEST_DISALLOW_COPY_AND_ASSIGN_(FactoryValueProducer);
};

static ValueProducer* producer_;
};

template <typename T>
class DefaultValue<T&> {
public:
static void Set(T& x) {  
address_ = &x;
}

static void Clear() { address_ = nullptr; }

static bool IsSet() { return address_ != nullptr; }

static bool Exists() {
return IsSet() || internal::BuiltInDefaultValue<T&>::Exists();
}

static T& Get() {
return address_ == nullptr ? internal::BuiltInDefaultValue<T&>::Get()
: *address_;
}

private:
static T* address_;
};

template <>
class DefaultValue<void> {
public:
static bool Exists() { return true; }
static void Get() {}
};

template <typename T>
typename DefaultValue<T>::ValueProducer* DefaultValue<T>::producer_ = nullptr;

template <typename T>
T* DefaultValue<T&>::address_ = nullptr;

template <typename F>
class ActionInterface {
public:
typedef typename internal::Function<F>::Result Result;
typedef typename internal::Function<F>::ArgumentTuple ArgumentTuple;

ActionInterface() {}
virtual ~ActionInterface() {}

virtual Result Perform(const ArgumentTuple& args) = 0;

private:
GTEST_DISALLOW_COPY_AND_ASSIGN_(ActionInterface);
};

template <typename F>
class Action {
struct ActionAdapter {
::std::shared_ptr<ActionInterface<F>> impl_;

template <typename... Args>
typename internal::Function<F>::Result operator()(Args&&... args) {
return impl_->Perform(
::std::forward_as_tuple(::std::forward<Args>(args)...));
}
};

public:
typedef typename internal::Function<F>::Result Result;
typedef typename internal::Function<F>::ArgumentTuple ArgumentTuple;

Action() {}

template <typename G,
typename IsCompatibleFunctor =
::std::is_constructible<::std::function<F>, G>,
typename IsNoArgsFunctor =
::std::is_constructible<::std::function<Result()>, G>,
typename = typename ::std::enable_if<internal::disjunction<
IsCompatibleFunctor, IsNoArgsFunctor>::value>::type>
Action(G&& fun) {  
Init(::std::forward<G>(fun), IsCompatibleFunctor());
}

explicit Action(ActionInterface<F>* impl)
: fun_(ActionAdapter{::std::shared_ptr<ActionInterface<F>>(impl)}) {}

template <typename Func>
explicit Action(const Action<Func>& action) : fun_(action.fun_) {}

bool IsDoDefault() const { return fun_ == nullptr; }

Result Perform(ArgumentTuple args) const {
if (IsDoDefault()) {
internal::IllegalDoDefault(__FILE__, __LINE__);
}
return internal::Apply(fun_, ::std::move(args));
}

private:
template <typename G>
friend class Action;

template <typename G>
void Init(G&& g, ::std::true_type) {
fun_ = ::std::forward<G>(g);
}

template <typename G>
void Init(G&& g, ::std::false_type) {
fun_ = IgnoreArgs<typename ::std::decay<G>::type>{::std::forward<G>(g)};
}

template <typename FunctionImpl>
struct IgnoreArgs {
template <typename... Args>
Result operator()(const Args&...) const {
return function_impl();
}

FunctionImpl function_impl;
};

::std::function<F> fun_;
};

template <typename Impl>
class PolymorphicAction {
public:
explicit PolymorphicAction(const Impl& impl) : impl_(impl) {}

template <typename F>
operator Action<F>() const {
return Action<F>(new MonomorphicImpl<F>(impl_));
}

private:
template <typename F>
class MonomorphicImpl : public ActionInterface<F> {
public:
typedef typename internal::Function<F>::Result Result;
typedef typename internal::Function<F>::ArgumentTuple ArgumentTuple;

explicit MonomorphicImpl(const Impl& impl) : impl_(impl) {}

Result Perform(const ArgumentTuple& args) override {
return impl_.template Perform<Result>(args);
}

private:
Impl impl_;
};

Impl impl_;
};

template <typename F>
Action<F> MakeAction(ActionInterface<F>* impl) {
return Action<F>(impl);
}

template <typename Impl>
inline PolymorphicAction<Impl> MakePolymorphicAction(const Impl& impl) {
return PolymorphicAction<Impl>(impl);
}

namespace internal {

template <typename T>
struct ByMoveWrapper {
explicit ByMoveWrapper(T value) : payload(std::move(value)) {}
T payload;
};

template <typename R>
class ReturnAction {
public:
explicit ReturnAction(R value) : value_(new R(std::move(value))) {}

template <typename F>
operator Action<F>() const {  
typedef typename Function<F>::Result Result;
GTEST_COMPILE_ASSERT_(
!std::is_reference<Result>::value,
use_ReturnRef_instead_of_Return_to_return_a_reference);
static_assert(!std::is_void<Result>::value,
"Can't use Return() on an action expected to return `void`.");
return Action<F>(new Impl<R, F>(value_));
}

private:
template <typename R_, typename F>
class Impl : public ActionInterface<F> {
public:
typedef typename Function<F>::Result Result;
typedef typename Function<F>::ArgumentTuple ArgumentTuple;

explicit Impl(const std::shared_ptr<R>& value)
: value_before_cast_(*value),
value_(ImplicitCast_<Result>(value_before_cast_)) {}

Result Perform(const ArgumentTuple&) override { return value_; }

private:
GTEST_COMPILE_ASSERT_(!std::is_reference<Result>::value,
Result_cannot_be_a_reference_type);
R value_before_cast_;
Result value_;

GTEST_DISALLOW_COPY_AND_ASSIGN_(Impl);
};

template <typename R_, typename F>
class Impl<ByMoveWrapper<R_>, F> : public ActionInterface<F> {
public:
typedef typename Function<F>::Result Result;
typedef typename Function<F>::ArgumentTuple ArgumentTuple;

explicit Impl(const std::shared_ptr<R>& wrapper)
: performed_(false), wrapper_(wrapper) {}

Result Perform(const ArgumentTuple&) override {
GTEST_CHECK_(!performed_)
<< "A ByMove() action should only be performed once.";
performed_ = true;
return std::move(wrapper_->payload);
}

private:
bool performed_;
const std::shared_ptr<R> wrapper_;
};

const std::shared_ptr<R> value_;
};

class ReturnNullAction {
public:
template <typename Result, typename ArgumentTuple>
static Result Perform(const ArgumentTuple&) {
return nullptr;
}
};

class ReturnVoidAction {
public:
template <typename Result, typename ArgumentTuple>
static void Perform(const ArgumentTuple&) {
static_assert(std::is_void<Result>::value, "Result should be void.");
}
};

template <typename T>
class ReturnRefAction {
public:
explicit ReturnRefAction(T& ref) : ref_(ref) {}  

template <typename F>
operator Action<F>() const {
typedef typename Function<F>::Result Result;
GTEST_COMPILE_ASSERT_(std::is_reference<Result>::value,
use_Return_instead_of_ReturnRef_to_return_a_value);
return Action<F>(new Impl<F>(ref_));
}

private:
template <typename F>
class Impl : public ActionInterface<F> {
public:
typedef typename Function<F>::Result Result;
typedef typename Function<F>::ArgumentTuple ArgumentTuple;

explicit Impl(T& ref) : ref_(ref) {}  

Result Perform(const ArgumentTuple&) override { return ref_; }

private:
T& ref_;
};

T& ref_;
};

template <typename T>
class ReturnRefOfCopyAction {
public:
explicit ReturnRefOfCopyAction(const T& value) : value_(value) {}  

template <typename F>
operator Action<F>() const {
typedef typename Function<F>::Result Result;
GTEST_COMPILE_ASSERT_(
std::is_reference<Result>::value,
use_Return_instead_of_ReturnRefOfCopy_to_return_a_value);
return Action<F>(new Impl<F>(value_));
}

private:
template <typename F>
class Impl : public ActionInterface<F> {
public:
typedef typename Function<F>::Result Result;
typedef typename Function<F>::ArgumentTuple ArgumentTuple;

explicit Impl(const T& value) : value_(value) {}  

Result Perform(const ArgumentTuple&) override { return value_; }

private:
T value_;
};

const T value_;
};

template <typename T>
class ReturnRoundRobinAction {
public:
explicit ReturnRoundRobinAction(std::vector<T> values) {
GTEST_CHECK_(!values.empty())
<< "ReturnRoundRobin requires at least one element.";
state_->values = std::move(values);
}

template <typename... Args>
T operator()(Args&&...) const {
return state_->Next();
}

private:
struct State {
T Next() {
T ret_val = values[i++];
if (i == values.size()) i = 0;
return ret_val;
}

std::vector<T> values;
size_t i = 0;
};
std::shared_ptr<State> state_ = std::make_shared<State>();
};

class DoDefaultAction {
public:
template <typename F>
operator Action<F>() const { return Action<F>(); }  
};

template <typename T1, typename T2>
class AssignAction {
public:
AssignAction(T1* ptr, T2 value) : ptr_(ptr), value_(value) {}

template <typename Result, typename ArgumentTuple>
void Perform(const ArgumentTuple& ) const {
*ptr_ = value_;
}

private:
T1* const ptr_;
const T2 value_;
};

#if !GTEST_OS_WINDOWS_MOBILE

template <typename T>
class SetErrnoAndReturnAction {
public:
SetErrnoAndReturnAction(int errno_value, T result)
: errno_(errno_value),
result_(result) {}
template <typename Result, typename ArgumentTuple>
Result Perform(const ArgumentTuple& ) const {
errno = errno_;
return result_;
}

private:
const int errno_;
const T result_;
};

#endif  

template <size_t N, typename A, typename = void>
struct SetArgumentPointeeAction {
A value;

template <typename... Args>
void operator()(const Args&... args) const {
*::std::get<N>(std::tie(args...)) = value;
}
};

template <class Class, typename MethodPtr>
struct InvokeMethodAction {
Class* const obj_ptr;
const MethodPtr method_ptr;

template <typename... Args>
auto operator()(Args&&... args) const
-> decltype((obj_ptr->*method_ptr)(std::forward<Args>(args)...)) {
return (obj_ptr->*method_ptr)(std::forward<Args>(args)...);
}
};

template <typename FunctionImpl>
struct InvokeWithoutArgsAction {
FunctionImpl function_impl;

template <typename... Args>
auto operator()(const Args&...) -> decltype(function_impl()) {
return function_impl();
}
};

template <class Class, typename MethodPtr>
struct InvokeMethodWithoutArgsAction {
Class* const obj_ptr;
const MethodPtr method_ptr;

using ReturnType =
decltype((std::declval<Class*>()->*std::declval<MethodPtr>())());

template <typename... Args>
ReturnType operator()(const Args&...) const {
return (obj_ptr->*method_ptr)();
}
};

template <typename A>
class IgnoreResultAction {
public:
explicit IgnoreResultAction(const A& action) : action_(action) {}

template <typename F>
operator Action<F>() const {
typedef typename internal::Function<F>::Result Result;

static_assert(std::is_void<Result>::value, "Result type should be void.");

return Action<F>(new Impl<F>(action_));
}

private:
template <typename F>
class Impl : public ActionInterface<F> {
public:
typedef typename internal::Function<F>::Result Result;
typedef typename internal::Function<F>::ArgumentTuple ArgumentTuple;

explicit Impl(const A& action) : action_(action) {}

void Perform(const ArgumentTuple& args) override {
action_.Perform(args);
}

private:
typedef typename internal::Function<F>::MakeResultIgnoredValue
OriginalFunction;

const Action<OriginalFunction> action_;
};

const A action_;
};

template <typename InnerAction, size_t... I>
struct WithArgsAction {
InnerAction action;

template <typename R, typename... Args>
operator Action<R(Args...)>() const {  
using TupleType = std::tuple<Args...>;
Action<R(typename std::tuple_element<I, TupleType>::type...)>
converted(action);

return [converted](Args... args) -> R {
return converted.Perform(std::forward_as_tuple(
std::get<I>(std::forward_as_tuple(std::forward<Args>(args)...))...));
};
}
};

template <typename... Actions>
struct DoAllAction {
private:
template <typename... Args, size_t... I>
std::vector<Action<void(Args...)>> Convert(IndexSequence<I...>) const {
return {std::get<I>(actions)...};
}

public:
std::tuple<Actions...> actions;

template <typename R, typename... Args>
operator Action<R(Args...)>() const {  
struct Op {
std::vector<Action<void(Args...)>> converted;
Action<R(Args...)> last;
R operator()(Args... args) const {
auto tuple_args = std::forward_as_tuple(std::forward<Args>(args)...);
for (auto& a : converted) {
a.Perform(tuple_args);
}
return last.Perform(tuple_args);
}
};
return Op{Convert<Args...>(MakeIndexSequence<sizeof...(Actions) - 1>()),
std::get<sizeof...(Actions) - 1>(actions)};
}
};

}  

typedef internal::IgnoredValue Unused;

template <typename... Action>
internal::DoAllAction<typename std::decay<Action>::type...> DoAll(
Action&&... action) {
return {std::forward_as_tuple(std::forward<Action>(action)...)};
}

template <size_t k, typename InnerAction>
internal::WithArgsAction<typename std::decay<InnerAction>::type, k>
WithArg(InnerAction&& action) {
return {std::forward<InnerAction>(action)};
}

template <size_t k, size_t... ks, typename InnerAction>
internal::WithArgsAction<typename std::decay<InnerAction>::type, k, ks...>
WithArgs(InnerAction&& action) {
return {std::forward<InnerAction>(action)};
}

template <typename InnerAction>
internal::WithArgsAction<typename std::decay<InnerAction>::type>
WithoutArgs(InnerAction&& action) {
return {std::forward<InnerAction>(action)};
}

template <typename R>
internal::ReturnAction<R> Return(R value) {
return internal::ReturnAction<R>(std::move(value));
}

inline PolymorphicAction<internal::ReturnNullAction> ReturnNull() {
return MakePolymorphicAction(internal::ReturnNullAction());
}

inline PolymorphicAction<internal::ReturnVoidAction> Return() {
return MakePolymorphicAction(internal::ReturnVoidAction());
}

template <typename R>
inline internal::ReturnRefAction<R> ReturnRef(R& x) {  
return internal::ReturnRefAction<R>(x);
}

template <typename R, R* = nullptr>
internal::ReturnRefAction<R> ReturnRef(R&&) = delete;

template <typename R>
inline internal::ReturnRefOfCopyAction<R> ReturnRefOfCopy(const R& x) {
return internal::ReturnRefOfCopyAction<R>(x);
}

template <typename R>
internal::ByMoveWrapper<R> ByMove(R x) {
return internal::ByMoveWrapper<R>(std::move(x));
}

template <typename T>
internal::ReturnRoundRobinAction<T> ReturnRoundRobin(std::vector<T> vals) {
return internal::ReturnRoundRobinAction<T>(std::move(vals));
}

template <typename T>
internal::ReturnRoundRobinAction<T> ReturnRoundRobin(
std::initializer_list<T> vals) {
return internal::ReturnRoundRobinAction<T>(std::vector<T>(vals));
}

inline internal::DoDefaultAction DoDefault() {
return internal::DoDefaultAction();
}

template <size_t N, typename T>
internal::SetArgumentPointeeAction<N, T> SetArgPointee(T value) {
return {std::move(value)};
}

template <size_t N, typename T>
internal::SetArgumentPointeeAction<N, T> SetArgumentPointee(T value) {
return {std::move(value)};
}

template <typename T1, typename T2>
PolymorphicAction<internal::AssignAction<T1, T2> > Assign(T1* ptr, T2 val) {
return MakePolymorphicAction(internal::AssignAction<T1, T2>(ptr, val));
}

#if !GTEST_OS_WINDOWS_MOBILE

template <typename T>
PolymorphicAction<internal::SetErrnoAndReturnAction<T> >
SetErrnoAndReturn(int errval, T result) {
return MakePolymorphicAction(
internal::SetErrnoAndReturnAction<T>(errval, result));
}

#endif  


template <typename FunctionImpl>
typename std::decay<FunctionImpl>::type Invoke(FunctionImpl&& function_impl) {
return std::forward<FunctionImpl>(function_impl);
}

template <class Class, typename MethodPtr>
internal::InvokeMethodAction<Class, MethodPtr> Invoke(Class* obj_ptr,
MethodPtr method_ptr) {
return {obj_ptr, method_ptr};
}

template <typename FunctionImpl>
internal::InvokeWithoutArgsAction<typename std::decay<FunctionImpl>::type>
InvokeWithoutArgs(FunctionImpl function_impl) {
return {std::move(function_impl)};
}

template <class Class, typename MethodPtr>
internal::InvokeMethodWithoutArgsAction<Class, MethodPtr> InvokeWithoutArgs(
Class* obj_ptr, MethodPtr method_ptr) {
return {obj_ptr, method_ptr};
}

template <typename A>
inline internal::IgnoreResultAction<A> IgnoreResult(const A& an_action) {
return internal::IgnoreResultAction<A>(an_action);
}

template <typename T>
inline ::std::reference_wrapper<T> ByRef(T& l_value) {  
return ::std::reference_wrapper<T>(l_value);
}

namespace internal {

template <typename T, typename... Params>
struct ReturnNewAction {
T* operator()() const {
return internal::Apply(
[](const Params&... unpacked_params) {
return new T(unpacked_params...);
},
params);
}
std::tuple<Params...> params;
};

}  

template <typename T, typename... Params>
internal::ReturnNewAction<T, typename std::decay<Params>::type...> ReturnNew(
Params&&... params) {
return {std::forward_as_tuple(std::forward<Params>(params)...)};
}

namespace internal {

struct ExcessiveArg {};

template <typename Result, class Impl>
class ActionHelper {
public:
template <typename... Ts>
static Result Perform(Impl* impl, const std::tuple<Ts...>& args) {
static constexpr size_t kMaxArgs = sizeof...(Ts) <= 10 ? sizeof...(Ts) : 10;
return Apply(impl, args, MakeIndexSequence<kMaxArgs>{},
MakeIndexSequence<10 - kMaxArgs>{});
}

private:
template <typename... Ts, std::size_t... tuple_ids, std::size_t... rest_ids>
static Result Apply(Impl* impl, const std::tuple<Ts...>& args,
IndexSequence<tuple_ids...>, IndexSequence<rest_ids...>) {
return impl->template gmock_PerformImpl<
typename std::tuple_element<tuple_ids, std::tuple<Ts...>>::type...>(
args, std::get<tuple_ids>(args)...,
((void)rest_ids, ExcessiveArg())...);
}
};

template <typename Derived>
class ActionImpl {
public:
ActionImpl() = default;

template <typename F>
operator ::testing::Action<F>() const {  
return ::testing::Action<F>(new typename Derived::template gmock_Impl<F>());
}
};

template <template <typename...> class Derived, typename... Ts>
class ActionImpl<Derived<Ts...>> {
public:
explicit ActionImpl(Ts... params) : params_(std::forward<Ts>(params)...) {}

template <typename F>
operator ::testing::Action<F>() const {  
return Apply<F>(MakeIndexSequence<sizeof...(Ts)>{});
}

private:
template <typename F, std::size_t... tuple_ids>
::testing::Action<F> Apply(IndexSequence<tuple_ids...>) const {
return ::testing::Action<F>(new
typename Derived<Ts...>::template gmock_Impl<F>(
std::get<tuple_ids>(params_)...));
}

std::tuple<Ts...> params_;
};

template <typename F, typename... Args>
auto InvokeArgument(F f, Args... args) -> decltype(f(args...)) {
return f(args...);
}

#define GMOCK_INTERNAL_ARG_UNUSED(i, data, el) \
, const arg##i##_type& arg##i GTEST_ATTRIBUTE_UNUSED_
#define GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_                 \
const args_type& args GTEST_ATTRIBUTE_UNUSED_ GMOCK_PP_REPEAT( \
GMOCK_INTERNAL_ARG_UNUSED, , 10)

#define GMOCK_INTERNAL_ARG(i, data, el) , const arg##i##_type& arg##i
#define GMOCK_ACTION_ARG_TYPES_AND_NAMES_ \
const args_type& args GMOCK_PP_REPEAT(GMOCK_INTERNAL_ARG, , 10)

#define GMOCK_INTERNAL_TEMPLATE_ARG(i, data, el) , typename arg##i##_type
#define GMOCK_ACTION_TEMPLATE_ARGS_NAMES_ \
GMOCK_PP_TAIL(GMOCK_PP_REPEAT(GMOCK_INTERNAL_TEMPLATE_ARG, , 10))

#define GMOCK_INTERNAL_TYPENAME_PARAM(i, data, param) , typename param##_type
#define GMOCK_ACTION_TYPENAME_PARAMS_(params) \
GMOCK_PP_TAIL(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_TYPENAME_PARAM, , params))

#define GMOCK_INTERNAL_TYPE_PARAM(i, data, param) , param##_type
#define GMOCK_ACTION_TYPE_PARAMS_(params) \
GMOCK_PP_TAIL(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_TYPE_PARAM, , params))

#define GMOCK_INTERNAL_TYPE_GVALUE_PARAM(i, data, param) \
, param##_type gmock_p##i
#define GMOCK_ACTION_TYPE_GVALUE_PARAMS_(params) \
GMOCK_PP_TAIL(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_TYPE_GVALUE_PARAM, , params))

#define GMOCK_INTERNAL_GVALUE_PARAM(i, data, param) \
, std::forward<param##_type>(gmock_p##i)
#define GMOCK_ACTION_GVALUE_PARAMS_(params) \
GMOCK_PP_TAIL(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_GVALUE_PARAM, , params))

#define GMOCK_INTERNAL_INIT_PARAM(i, data, param) \
, param(::std::forward<param##_type>(gmock_p##i))
#define GMOCK_ACTION_INIT_PARAMS_(params) \
GMOCK_PP_TAIL(GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_INIT_PARAM, , params))

#define GMOCK_INTERNAL_FIELD_PARAM(i, data, param) param##_type param;
#define GMOCK_ACTION_FIELD_PARAMS_(params) \
GMOCK_PP_FOR_EACH(GMOCK_INTERNAL_FIELD_PARAM, , params)

#define GMOCK_INTERNAL_ACTION(name, full_name, params)                        \
template <GMOCK_ACTION_TYPENAME_PARAMS_(params)>                            \
class full_name : public ::testing::internal::ActionImpl<                   \
full_name<GMOCK_ACTION_TYPE_PARAMS_(params)>> {       \
using base_type = ::testing::internal::ActionImpl<full_name>;             \
\
public:                                                                    \
using base_type::base_type;                                               \
template <typename F>                                                     \
class gmock_Impl : public ::testing::ActionInterface<F> {                 \
public:                                                                  \
typedef F function_type;                                                \
typedef typename ::testing::internal::Function<F>::Result return_type;  \
typedef                                                                 \
typename ::testing::internal::Function<F>::ArgumentTuple args_type; \
explicit gmock_Impl(GMOCK_ACTION_TYPE_GVALUE_PARAMS_(params))           \
: GMOCK_ACTION_INIT_PARAMS_(params) {}                              \
return_type Perform(const args_type& args) override {                   \
return ::testing::internal::ActionHelper<return_type,                 \
gmock_Impl>::Perform(this,   \
args);  \
}                                                                       \
template <GMOCK_ACTION_TEMPLATE_ARGS_NAMES_>                            \
return_type gmock_PerformImpl(GMOCK_ACTION_ARG_TYPES_AND_NAMES_) const; \
GMOCK_ACTION_FIELD_PARAMS_(params)                                      \
};                                                                        \
};                                                                          \
template <GMOCK_ACTION_TYPENAME_PARAMS_(params)>                            \
inline full_name<GMOCK_ACTION_TYPE_PARAMS_(params)> name(                   \
GMOCK_ACTION_TYPE_GVALUE_PARAMS_(params)) {                             \
return full_name<GMOCK_ACTION_TYPE_PARAMS_(params)>(                      \
GMOCK_ACTION_GVALUE_PARAMS_(params));                                 \
}                                                                           \
template <GMOCK_ACTION_TYPENAME_PARAMS_(params)>                            \
template <typename F>                                                       \
template <GMOCK_ACTION_TEMPLATE_ARGS_NAMES_>                                \
typename ::testing::internal::Function<F>::Result                           \
full_name<GMOCK_ACTION_TYPE_PARAMS_(params)>::gmock_Impl<               \
F>::gmock_PerformImpl(GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_)     \
const

}  

#define ACTION(name)                                                          \
class name##Action : public ::testing::internal::ActionImpl<name##Action> { \
using base_type = ::testing::internal::ActionImpl<name##Action>;          \
\
public:                                                                    \
using base_type::base_type;                                               \
template <typename F>                                                     \
class gmock_Impl : public ::testing::ActionInterface<F> {                 \
public:                                                                  \
typedef F function_type;                                                \
typedef typename ::testing::internal::Function<F>::Result return_type;  \
typedef                                                                 \
typename ::testing::internal::Function<F>::ArgumentTuple args_type; \
gmock_Impl() {}                                                         \
return_type Perform(const args_type& args) override {                   \
return ::testing::internal::ActionHelper<return_type,                 \
gmock_Impl>::Perform(this,   \
args);  \
}                                                                       \
template <GMOCK_ACTION_TEMPLATE_ARGS_NAMES_>                            \
return_type gmock_PerformImpl(GMOCK_ACTION_ARG_TYPES_AND_NAMES_) const; \
};                                                                        \
};                                                                          \
inline name##Action name() { return name##Action(); }                       \
template <typename F>                                                       \
template <GMOCK_ACTION_TEMPLATE_ARGS_NAMES_>                                \
typename ::testing::internal::Function<F>::Result                           \
name##Action::gmock_Impl<F>::gmock_PerformImpl(                         \
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP, (__VA_ARGS__))

#define ACTION_P2(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP2, (__VA_ARGS__))

#define ACTION_P3(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP3, (__VA_ARGS__))

#define ACTION_P4(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP4, (__VA_ARGS__))

#define ACTION_P5(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP5, (__VA_ARGS__))

#define ACTION_P6(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP6, (__VA_ARGS__))

#define ACTION_P7(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP7, (__VA_ARGS__))

#define ACTION_P8(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP8, (__VA_ARGS__))

#define ACTION_P9(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP9, (__VA_ARGS__))

#define ACTION_P10(name, ...) \
GMOCK_INTERNAL_ACTION(name, name##ActionP10, (__VA_ARGS__))

}  

#ifdef _MSC_VER
# pragma warning(pop)
#endif


#endif  
