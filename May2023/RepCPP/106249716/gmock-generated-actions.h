


#ifndef GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_ACTIONS_H_
#define GMOCK_INCLUDE_GMOCK_GMOCK_GENERATED_ACTIONS_H_

#include "gmock/gmock-actions.h"
#include "gmock/internal/gmock-port.h"

namespace testing {
namespace internal {

template <typename Result, typename ArgumentTuple>
class InvokeHelper;

template <typename R>
class InvokeHelper<R, ::testing::tuple<> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<>&) {
return function();
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<>&) {
return (obj_ptr->*method_ptr)();
}
};

template <typename R, typename A1>
class InvokeHelper<R, ::testing::tuple<A1> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1>& args) {
return function(get<0>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1>& args) {
return (obj_ptr->*method_ptr)(get<0>(args));
}
};

template <typename R, typename A1, typename A2>
class InvokeHelper<R, ::testing::tuple<A1, A2> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2>& args) {
return function(get<0>(args), get<1>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args));
}
};

template <typename R, typename A1, typename A2, typename A3>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3>& args) {
return function(get<0>(args), get<1>(args), get<2>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3,
A4>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4,
typename A5>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4, A5> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3, A4,
A5>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args), get<4>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4, A5>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args), get<4>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4, A5, A6> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3, A4, A5,
A6>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args), get<4>(args), get<5>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4, A5, A6>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args), get<4>(args), get<5>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4, A5, A6, A7> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3, A4, A5,
A6, A7>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args), get<4>(args), get<5>(args), get<6>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4, A5, A6,
A7>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args), get<4>(args), get<5>(args),
get<6>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7, typename A8>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4, A5, A6, A7, A8> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3, A4, A5,
A6, A7, A8>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args), get<4>(args), get<5>(args), get<6>(args),
get<7>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4, A5, A6, A7,
A8>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args), get<4>(args), get<5>(args),
get<6>(args), get<7>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7, typename A8, typename A9>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4, A5, A6, A7, A8, A9> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3, A4, A5,
A6, A7, A8, A9>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args), get<4>(args), get<5>(args), get<6>(args),
get<7>(args), get<8>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4, A5, A6, A7, A8,
A9>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args), get<4>(args), get<5>(args),
get<6>(args), get<7>(args), get<8>(args));
}
};

template <typename R, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7, typename A8, typename A9,
typename A10>
class InvokeHelper<R, ::testing::tuple<A1, A2, A3, A4, A5, A6, A7, A8, A9,
A10> > {
public:
template <typename Function>
static R Invoke(Function function, const ::testing::tuple<A1, A2, A3, A4, A5,
A6, A7, A8, A9, A10>& args) {
return function(get<0>(args), get<1>(args), get<2>(args),
get<3>(args), get<4>(args), get<5>(args), get<6>(args),
get<7>(args), get<8>(args), get<9>(args));
}

template <class Class, typename MethodPtr>
static R InvokeMethod(Class* obj_ptr,
MethodPtr method_ptr,
const ::testing::tuple<A1, A2, A3, A4, A5, A6, A7, A8,
A9, A10>& args) {
return (obj_ptr->*method_ptr)(get<0>(args), get<1>(args),
get<2>(args), get<3>(args), get<4>(args), get<5>(args),
get<6>(args), get<7>(args), get<8>(args), get<9>(args));
}
};

#define GMOCK_FIELD_(Tuple, N) \
typename ::testing::tuple_element<N, Tuple>::type


template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4, int k5, int k6, int k7, int k8, int k9, int k10>
class SelectArgs {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4), GMOCK_FIELD_(ArgumentTuple, k5),
GMOCK_FIELD_(ArgumentTuple, k6), GMOCK_FIELD_(ArgumentTuple, k7),
GMOCK_FIELD_(ArgumentTuple, k8), GMOCK_FIELD_(ArgumentTuple, k9),
GMOCK_FIELD_(ArgumentTuple, k10));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args), get<k5>(args), get<k6>(args), get<k7>(args),
get<k8>(args), get<k9>(args), get<k10>(args));
}
};

template <typename Result, typename ArgumentTuple>
class SelectArgs<Result, ArgumentTuple,
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1> {
public:
typedef Result type();
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& ) {
return SelectedArgs();
}
};

template <typename Result, typename ArgumentTuple, int k1>
class SelectArgs<Result, ArgumentTuple,
k1, -1, -1, -1, -1, -1, -1, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2>
class SelectArgs<Result, ArgumentTuple,
k1, k2, -1, -1, -1, -1, -1, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, -1, -1, -1, -1, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, -1, -1, -1, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4, int k5>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, k5, -1, -1, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4), GMOCK_FIELD_(ArgumentTuple, k5));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args), get<k5>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4, int k5, int k6>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, k5, k6, -1, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4), GMOCK_FIELD_(ArgumentTuple, k5),
GMOCK_FIELD_(ArgumentTuple, k6));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args), get<k5>(args), get<k6>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4, int k5, int k6, int k7>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, k5, k6, k7, -1, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4), GMOCK_FIELD_(ArgumentTuple, k5),
GMOCK_FIELD_(ArgumentTuple, k6), GMOCK_FIELD_(ArgumentTuple, k7));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args), get<k5>(args), get<k6>(args), get<k7>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4, int k5, int k6, int k7, int k8>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, k5, k6, k7, k8, -1, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4), GMOCK_FIELD_(ArgumentTuple, k5),
GMOCK_FIELD_(ArgumentTuple, k6), GMOCK_FIELD_(ArgumentTuple, k7),
GMOCK_FIELD_(ArgumentTuple, k8));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args), get<k5>(args), get<k6>(args), get<k7>(args),
get<k8>(args));
}
};

template <typename Result, typename ArgumentTuple, int k1, int k2, int k3,
int k4, int k5, int k6, int k7, int k8, int k9>
class SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, k5, k6, k7, k8, k9, -1> {
public:
typedef Result type(GMOCK_FIELD_(ArgumentTuple, k1),
GMOCK_FIELD_(ArgumentTuple, k2), GMOCK_FIELD_(ArgumentTuple, k3),
GMOCK_FIELD_(ArgumentTuple, k4), GMOCK_FIELD_(ArgumentTuple, k5),
GMOCK_FIELD_(ArgumentTuple, k6), GMOCK_FIELD_(ArgumentTuple, k7),
GMOCK_FIELD_(ArgumentTuple, k8), GMOCK_FIELD_(ArgumentTuple, k9));
typedef typename Function<type>::ArgumentTuple SelectedArgs;
static SelectedArgs Select(const ArgumentTuple& args) {
return SelectedArgs(get<k1>(args), get<k2>(args), get<k3>(args),
get<k4>(args), get<k5>(args), get<k6>(args), get<k7>(args),
get<k8>(args), get<k9>(args));
}
};

#undef GMOCK_FIELD_

template <typename InnerAction, int k1 = -1, int k2 = -1, int k3 = -1,
int k4 = -1, int k5 = -1, int k6 = -1, int k7 = -1, int k8 = -1,
int k9 = -1, int k10 = -1>
class WithArgsAction {
public:
explicit WithArgsAction(const InnerAction& action) : action_(action) {}

template <typename F>
operator Action<F>() const { return MakeAction(new Impl<F>(action_)); }

private:
template <typename F>
class Impl : public ActionInterface<F> {
public:
typedef typename Function<F>::Result Result;
typedef typename Function<F>::ArgumentTuple ArgumentTuple;

explicit Impl(const InnerAction& action) : action_(action) {}

virtual Result Perform(const ArgumentTuple& args) {
return action_.Perform(SelectArgs<Result, ArgumentTuple, k1, k2, k3, k4,
k5, k6, k7, k8, k9, k10>::Select(args));
}

private:
typedef typename SelectArgs<Result, ArgumentTuple,
k1, k2, k3, k4, k5, k6, k7, k8, k9, k10>::type InnerFunctionType;

Action<InnerFunctionType> action_;
};

const InnerAction action_;

GTEST_DISALLOW_ASSIGN_(WithArgsAction);
};

struct ExcessiveArg {};

template <typename Result, class Impl>
class ActionHelper {
public:
static Result Perform(Impl* impl, const ::testing::tuple<>& args) {
return impl->template gmock_PerformImpl<>(args, ExcessiveArg(),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0>
static Result Perform(Impl* impl, const ::testing::tuple<A0>& args) {
return impl->template gmock_PerformImpl<A0>(args, get<0>(args),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1>& args) {
return impl->template gmock_PerformImpl<A0, A1>(args, get<0>(args),
get<1>(args), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2>(args, get<0>(args),
get<1>(args), get<2>(args), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2,
A3>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3>(args, get<0>(args),
get<1>(args), get<2>(args), get<3>(args), ExcessiveArg(),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2, A3,
A4>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3, A4>(args,
get<0>(args), get<1>(args), get<2>(args), get<3>(args), get<4>(args),
ExcessiveArg(), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
typename A5>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2, A3, A4,
A5>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3, A4, A5>(args,
get<0>(args), get<1>(args), get<2>(args), get<3>(args), get<4>(args),
get<5>(args), ExcessiveArg(), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2, A3, A4,
A5, A6>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3, A4, A5, A6>(args,
get<0>(args), get<1>(args), get<2>(args), get<3>(args), get<4>(args),
get<5>(args), get<6>(args), ExcessiveArg(), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2, A3, A4,
A5, A6, A7>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3, A4, A5, A6,
A7>(args, get<0>(args), get<1>(args), get<2>(args), get<3>(args),
get<4>(args), get<5>(args), get<6>(args), get<7>(args), ExcessiveArg(),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7, typename A8>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2, A3, A4,
A5, A6, A7, A8>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3, A4, A5, A6, A7,
A8>(args, get<0>(args), get<1>(args), get<2>(args), get<3>(args),
get<4>(args), get<5>(args), get<6>(args), get<7>(args), get<8>(args),
ExcessiveArg());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
typename A5, typename A6, typename A7, typename A8, typename A9>
static Result Perform(Impl* impl, const ::testing::tuple<A0, A1, A2, A3, A4,
A5, A6, A7, A8, A9>& args) {
return impl->template gmock_PerformImpl<A0, A1, A2, A3, A4, A5, A6, A7, A8,
A9>(args, get<0>(args), get<1>(args), get<2>(args), get<3>(args),
get<4>(args), get<5>(args), get<6>(args), get<7>(args), get<8>(args),
get<9>(args));
}
};

}  


template <int k1, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1>(action);
}

template <int k1, int k2, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2>(action);
}

template <int k1, int k2, int k3, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3>(action);
}

template <int k1, int k2, int k3, int k4, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4>(action);
}

template <int k1, int k2, int k3, int k4, int k5, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5>(action);
}

template <int k1, int k2, int k3, int k4, int k5, int k6, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6>(action);
}

template <int k1, int k2, int k3, int k4, int k5, int k6, int k7,
typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6,
k7>(action);
}

template <int k1, int k2, int k3, int k4, int k5, int k6, int k7, int k8,
typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7, k8>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7,
k8>(action);
}

template <int k1, int k2, int k3, int k4, int k5, int k6, int k7, int k8,
int k9, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7, k8, k9>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7, k8,
k9>(action);
}

template <int k1, int k2, int k3, int k4, int k5, int k6, int k7, int k8,
int k9, int k10, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7, k8,
k9, k10>
WithArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k1, k2, k3, k4, k5, k6, k7, k8,
k9, k10>(action);
}

template <typename Action1, typename Action2>
inline internal::DoBothAction<Action1, Action2>
DoAll(Action1 a1, Action2 a2) {
return internal::DoBothAction<Action1, Action2>(a1, a2);
}

template <typename Action1, typename Action2, typename Action3>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
Action3> >
DoAll(Action1 a1, Action2 a2, Action3 a3) {
return DoAll(a1, DoAll(a2, a3));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, Action4> > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4) {
return DoAll(a1, DoAll(a2, a3, a4));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4, typename Action5>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, internal::DoBothAction<Action4,
Action5> > > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4, Action5 a5) {
return DoAll(a1, DoAll(a2, a3, a4, a5));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4, typename Action5, typename Action6>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, internal::DoBothAction<Action4,
internal::DoBothAction<Action5, Action6> > > > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4, Action5 a5, Action6 a6) {
return DoAll(a1, DoAll(a2, a3, a4, a5, a6));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4, typename Action5, typename Action6, typename Action7>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, internal::DoBothAction<Action4,
internal::DoBothAction<Action5, internal::DoBothAction<Action6,
Action7> > > > > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4, Action5 a5, Action6 a6,
Action7 a7) {
return DoAll(a1, DoAll(a2, a3, a4, a5, a6, a7));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4, typename Action5, typename Action6, typename Action7,
typename Action8>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, internal::DoBothAction<Action4,
internal::DoBothAction<Action5, internal::DoBothAction<Action6,
internal::DoBothAction<Action7, Action8> > > > > > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4, Action5 a5, Action6 a6,
Action7 a7, Action8 a8) {
return DoAll(a1, DoAll(a2, a3, a4, a5, a6, a7, a8));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4, typename Action5, typename Action6, typename Action7,
typename Action8, typename Action9>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, internal::DoBothAction<Action4,
internal::DoBothAction<Action5, internal::DoBothAction<Action6,
internal::DoBothAction<Action7, internal::DoBothAction<Action8,
Action9> > > > > > > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4, Action5 a5, Action6 a6,
Action7 a7, Action8 a8, Action9 a9) {
return DoAll(a1, DoAll(a2, a3, a4, a5, a6, a7, a8, a9));
}

template <typename Action1, typename Action2, typename Action3,
typename Action4, typename Action5, typename Action6, typename Action7,
typename Action8, typename Action9, typename Action10>
inline internal::DoBothAction<Action1, internal::DoBothAction<Action2,
internal::DoBothAction<Action3, internal::DoBothAction<Action4,
internal::DoBothAction<Action5, internal::DoBothAction<Action6,
internal::DoBothAction<Action7, internal::DoBothAction<Action8,
internal::DoBothAction<Action9, Action10> > > > > > > > >
DoAll(Action1 a1, Action2 a2, Action3 a3, Action4 a4, Action5 a5, Action6 a6,
Action7 a7, Action8 a8, Action9 a9, Action10 a10) {
return DoAll(a1, DoAll(a2, a3, a4, a5, a6, a7, a8, a9, a10));
}

}  


#define GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_\
const args_type& args GTEST_ATTRIBUTE_UNUSED_, \
arg0_type arg0 GTEST_ATTRIBUTE_UNUSED_, \
arg1_type arg1 GTEST_ATTRIBUTE_UNUSED_, \
arg2_type arg2 GTEST_ATTRIBUTE_UNUSED_, \
arg3_type arg3 GTEST_ATTRIBUTE_UNUSED_, \
arg4_type arg4 GTEST_ATTRIBUTE_UNUSED_, \
arg5_type arg5 GTEST_ATTRIBUTE_UNUSED_, \
arg6_type arg6 GTEST_ATTRIBUTE_UNUSED_, \
arg7_type arg7 GTEST_ATTRIBUTE_UNUSED_, \
arg8_type arg8 GTEST_ATTRIBUTE_UNUSED_, \
arg9_type arg9 GTEST_ATTRIBUTE_UNUSED_


#define GMOCK_INTERNAL_DECL_HAS_1_TEMPLATE_PARAMS(kind0, name0) kind0 name0
#define GMOCK_INTERNAL_DECL_HAS_2_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1) kind0 name0, kind1 name1
#define GMOCK_INTERNAL_DECL_HAS_3_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2) kind0 name0, kind1 name1, kind2 name2
#define GMOCK_INTERNAL_DECL_HAS_4_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3) kind0 name0, kind1 name1, kind2 name2, \
kind3 name3
#define GMOCK_INTERNAL_DECL_HAS_5_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4) kind0 name0, kind1 name1, \
kind2 name2, kind3 name3, kind4 name4
#define GMOCK_INTERNAL_DECL_HAS_6_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5) kind0 name0, \
kind1 name1, kind2 name2, kind3 name3, kind4 name4, kind5 name5
#define GMOCK_INTERNAL_DECL_HAS_7_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6) kind0 name0, kind1 name1, kind2 name2, kind3 name3, kind4 name4, \
kind5 name5, kind6 name6
#define GMOCK_INTERNAL_DECL_HAS_8_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7) kind0 name0, kind1 name1, kind2 name2, kind3 name3, \
kind4 name4, kind5 name5, kind6 name6, kind7 name7
#define GMOCK_INTERNAL_DECL_HAS_9_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7, kind8, name8) kind0 name0, kind1 name1, kind2 name2, \
kind3 name3, kind4 name4, kind5 name5, kind6 name6, kind7 name7, \
kind8 name8
#define GMOCK_INTERNAL_DECL_HAS_10_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1, kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6, kind7, name7, kind8, name8, kind9, name9) kind0 name0, \
kind1 name1, kind2 name2, kind3 name3, kind4 name4, kind5 name5, \
kind6 name6, kind7 name7, kind8 name8, kind9 name9

#define GMOCK_INTERNAL_LIST_HAS_1_TEMPLATE_PARAMS(kind0, name0) name0
#define GMOCK_INTERNAL_LIST_HAS_2_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1) name0, name1
#define GMOCK_INTERNAL_LIST_HAS_3_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2) name0, name1, name2
#define GMOCK_INTERNAL_LIST_HAS_4_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3) name0, name1, name2, name3
#define GMOCK_INTERNAL_LIST_HAS_5_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4) name0, name1, name2, name3, \
name4
#define GMOCK_INTERNAL_LIST_HAS_6_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5) name0, name1, \
name2, name3, name4, name5
#define GMOCK_INTERNAL_LIST_HAS_7_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6) name0, name1, name2, name3, name4, name5, name6
#define GMOCK_INTERNAL_LIST_HAS_8_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7) name0, name1, name2, name3, name4, name5, name6, name7
#define GMOCK_INTERNAL_LIST_HAS_9_TEMPLATE_PARAMS(kind0, name0, kind1, name1, \
kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, name6, \
kind7, name7, kind8, name8) name0, name1, name2, name3, name4, name5, \
name6, name7, name8
#define GMOCK_INTERNAL_LIST_HAS_10_TEMPLATE_PARAMS(kind0, name0, kind1, \
name1, kind2, name2, kind3, name3, kind4, name4, kind5, name5, kind6, \
name6, kind7, name7, kind8, name8, kind9, name9) name0, name1, name2, \
name3, name4, name5, name6, name7, name8, name9

#define GMOCK_INTERNAL_DECL_TYPE_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_DECL_TYPE_AND_1_VALUE_PARAMS(p0) , typename p0##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_2_VALUE_PARAMS(p0, p1) , \
typename p0##_type, typename p1##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_3_VALUE_PARAMS(p0, p1, p2) , \
typename p0##_type, typename p1##_type, typename p2##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_4_VALUE_PARAMS(p0, p1, p2, p3) , \
typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) , \
typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) , \
typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) , typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7) , typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8) , typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type
#define GMOCK_INTERNAL_DECL_TYPE_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8, p9) , typename p0##_type, typename p1##_type, \
typename p2##_type, typename p3##_type, typename p4##_type, \
typename p5##_type, typename p6##_type, typename p7##_type, \
typename p8##_type, typename p9##_type

#define GMOCK_INTERNAL_INIT_AND_0_VALUE_PARAMS()\
()
#define GMOCK_INTERNAL_INIT_AND_1_VALUE_PARAMS(p0)\
(p0##_type gmock_p0) : p0(gmock_p0)
#define GMOCK_INTERNAL_INIT_AND_2_VALUE_PARAMS(p0, p1)\
(p0##_type gmock_p0, p1##_type gmock_p1) : p0(gmock_p0), p1(gmock_p1)
#define GMOCK_INTERNAL_INIT_AND_3_VALUE_PARAMS(p0, p1, p2)\
(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2)
#define GMOCK_INTERNAL_INIT_AND_4_VALUE_PARAMS(p0, p1, p2, p3)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3)
#define GMOCK_INTERNAL_INIT_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4) : p0(gmock_p0), p1(gmock_p1), \
p2(gmock_p2), p3(gmock_p3), p4(gmock_p4)
#define GMOCK_INTERNAL_INIT_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5)
#define GMOCK_INTERNAL_INIT_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6)
#define GMOCK_INTERNAL_INIT_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7) : p0(gmock_p0), p1(gmock_p1), \
p2(gmock_p2), p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), \
p7(gmock_p7)
#define GMOCK_INTERNAL_INIT_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7, \
p8##_type gmock_p8) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), p7(gmock_p7), \
p8(gmock_p8)
#define GMOCK_INTERNAL_INIT_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9)\
(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7, p8##_type gmock_p8, \
p9##_type gmock_p9) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), p7(gmock_p7), \
p8(gmock_p8), p9(gmock_p9)

#define GMOCK_INTERNAL_DEFN_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_DEFN_AND_1_VALUE_PARAMS(p0) p0##_type p0;
#define GMOCK_INTERNAL_DEFN_AND_2_VALUE_PARAMS(p0, p1) p0##_type p0; \
p1##_type p1;
#define GMOCK_INTERNAL_DEFN_AND_3_VALUE_PARAMS(p0, p1, p2) p0##_type p0; \
p1##_type p1; p2##_type p2;
#define GMOCK_INTERNAL_DEFN_AND_4_VALUE_PARAMS(p0, p1, p2, p3) p0##_type p0; \
p1##_type p1; p2##_type p2; p3##_type p3;
#define GMOCK_INTERNAL_DEFN_AND_5_VALUE_PARAMS(p0, p1, p2, p3, \
p4) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4;
#define GMOCK_INTERNAL_DEFN_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, \
p5) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4; \
p5##_type p5;
#define GMOCK_INTERNAL_DEFN_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4; \
p5##_type p5; p6##_type p6;
#define GMOCK_INTERNAL_DEFN_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; p4##_type p4; \
p5##_type p5; p6##_type p6; p7##_type p7;
#define GMOCK_INTERNAL_DEFN_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; \
p4##_type p4; p5##_type p5; p6##_type p6; p7##_type p7; p8##_type p8;
#define GMOCK_INTERNAL_DEFN_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) p0##_type p0; p1##_type p1; p2##_type p2; p3##_type p3; \
p4##_type p4; p5##_type p5; p6##_type p6; p7##_type p7; p8##_type p8; \
p9##_type p9;

#define GMOCK_INTERNAL_LIST_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_LIST_AND_1_VALUE_PARAMS(p0) p0
#define GMOCK_INTERNAL_LIST_AND_2_VALUE_PARAMS(p0, p1) p0, p1
#define GMOCK_INTERNAL_LIST_AND_3_VALUE_PARAMS(p0, p1, p2) p0, p1, p2
#define GMOCK_INTERNAL_LIST_AND_4_VALUE_PARAMS(p0, p1, p2, p3) p0, p1, p2, p3
#define GMOCK_INTERNAL_LIST_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) p0, p1, \
p2, p3, p4
#define GMOCK_INTERNAL_LIST_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) p0, \
p1, p2, p3, p4, p5
#define GMOCK_INTERNAL_LIST_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) p0, p1, p2, p3, p4, p5, p6
#define GMOCK_INTERNAL_LIST_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) p0, p1, p2, p3, p4, p5, p6, p7
#define GMOCK_INTERNAL_LIST_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) p0, p1, p2, p3, p4, p5, p6, p7, p8
#define GMOCK_INTERNAL_LIST_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) p0, p1, p2, p3, p4, p5, p6, p7, p8, p9

#define GMOCK_INTERNAL_LIST_TYPE_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_LIST_TYPE_AND_1_VALUE_PARAMS(p0) , p0##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_2_VALUE_PARAMS(p0, p1) , p0##_type, \
p1##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_3_VALUE_PARAMS(p0, p1, p2) , p0##_type, \
p1##_type, p2##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_4_VALUE_PARAMS(p0, p1, p2, p3) , \
p0##_type, p1##_type, p2##_type, p3##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) , \
p0##_type, p1##_type, p2##_type, p3##_type, p4##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) , \
p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, p5##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, p5##_type, \
p6##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type, p8##_type
#define GMOCK_INTERNAL_LIST_TYPE_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6, p7, p8, p9) , p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type, p8##_type, p9##_type

#define GMOCK_INTERNAL_DECL_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_DECL_AND_1_VALUE_PARAMS(p0) p0##_type p0
#define GMOCK_INTERNAL_DECL_AND_2_VALUE_PARAMS(p0, p1) p0##_type p0, \
p1##_type p1
#define GMOCK_INTERNAL_DECL_AND_3_VALUE_PARAMS(p0, p1, p2) p0##_type p0, \
p1##_type p1, p2##_type p2
#define GMOCK_INTERNAL_DECL_AND_4_VALUE_PARAMS(p0, p1, p2, p3) p0##_type p0, \
p1##_type p1, p2##_type p2, p3##_type p3
#define GMOCK_INTERNAL_DECL_AND_5_VALUE_PARAMS(p0, p1, p2, p3, \
p4) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4
#define GMOCK_INTERNAL_DECL_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, \
p5) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, \
p5##_type p5
#define GMOCK_INTERNAL_DECL_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, \
p6) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, \
p5##_type p5, p6##_type p6
#define GMOCK_INTERNAL_DECL_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, \
p5##_type p5, p6##_type p6, p7##_type p7
#define GMOCK_INTERNAL_DECL_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4, p5##_type p5, p6##_type p6, p7##_type p7, p8##_type p8
#define GMOCK_INTERNAL_DECL_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4, p5##_type p5, p6##_type p6, p7##_type p7, p8##_type p8, \
p9##_type p9

#define GMOCK_INTERNAL_COUNT_AND_0_VALUE_PARAMS()
#define GMOCK_INTERNAL_COUNT_AND_1_VALUE_PARAMS(p0) P
#define GMOCK_INTERNAL_COUNT_AND_2_VALUE_PARAMS(p0, p1) P2
#define GMOCK_INTERNAL_COUNT_AND_3_VALUE_PARAMS(p0, p1, p2) P3
#define GMOCK_INTERNAL_COUNT_AND_4_VALUE_PARAMS(p0, p1, p2, p3) P4
#define GMOCK_INTERNAL_COUNT_AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4) P5
#define GMOCK_INTERNAL_COUNT_AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5) P6
#define GMOCK_INTERNAL_COUNT_AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6) P7
#define GMOCK_INTERNAL_COUNT_AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7) P8
#define GMOCK_INTERNAL_COUNT_AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8) P9
#define GMOCK_INTERNAL_COUNT_AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, \
p7, p8, p9) P10

#define GMOCK_ACTION_CLASS_(name, value_params)\
GTEST_CONCAT_TOKEN_(name##Action, GMOCK_INTERNAL_COUNT_##value_params)

#define ACTION_TEMPLATE(name, template_params, value_params)\
template <GMOCK_INTERNAL_DECL_##template_params\
GMOCK_INTERNAL_DECL_TYPE_##value_params>\
class GMOCK_ACTION_CLASS_(name, value_params) {\
public:\
explicit GMOCK_ACTION_CLASS_(name, value_params)\
GMOCK_INTERNAL_INIT_##value_params {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
explicit gmock_Impl GMOCK_INTERNAL_INIT_##value_params {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
GMOCK_INTERNAL_DEFN_##value_params\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(\
new gmock_Impl<F>(GMOCK_INTERNAL_LIST_##value_params));\
}\
GMOCK_INTERNAL_DEFN_##value_params\
private:\
GTEST_DISALLOW_ASSIGN_(GMOCK_ACTION_CLASS_(name, value_params));\
};\
template <GMOCK_INTERNAL_DECL_##template_params\
GMOCK_INTERNAL_DECL_TYPE_##value_params>\
inline GMOCK_ACTION_CLASS_(name, value_params)<\
GMOCK_INTERNAL_LIST_##template_params\
GMOCK_INTERNAL_LIST_TYPE_##value_params> name(\
GMOCK_INTERNAL_DECL_##value_params) {\
return GMOCK_ACTION_CLASS_(name, value_params)<\
GMOCK_INTERNAL_LIST_##template_params\
GMOCK_INTERNAL_LIST_TYPE_##value_params>(\
GMOCK_INTERNAL_LIST_##value_params);\
}\
template <GMOCK_INTERNAL_DECL_##template_params\
GMOCK_INTERNAL_DECL_TYPE_##value_params>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
GMOCK_ACTION_CLASS_(name, value_params)<\
GMOCK_INTERNAL_LIST_##template_params\
GMOCK_INTERNAL_LIST_TYPE_##value_params>::gmock_Impl<F>::\
gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION(name)\
class name##Action {\
public:\
name##Action() {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl() {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>());\
}\
private:\
GTEST_DISALLOW_ASSIGN_(name##Action);\
};\
inline name##Action name() {\
return name##Action();\
}\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##Action::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P(name, p0)\
template <typename p0##_type>\
class name##ActionP {\
public:\
explicit name##ActionP(p0##_type gmock_p0) : p0(gmock_p0) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
explicit gmock_Impl(p0##_type gmock_p0) : p0(gmock_p0) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0));\
}\
p0##_type p0;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP);\
};\
template <typename p0##_type>\
inline name##ActionP<p0##_type> name(p0##_type p0) {\
return name##ActionP<p0##_type>(p0);\
}\
template <typename p0##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP<p0##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P2(name, p0, p1)\
template <typename p0##_type, typename p1##_type>\
class name##ActionP2 {\
public:\
name##ActionP2(p0##_type gmock_p0, p1##_type gmock_p1) : p0(gmock_p0), \
p1(gmock_p1) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1) : p0(gmock_p0), \
p1(gmock_p1) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1));\
}\
p0##_type p0;\
p1##_type p1;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP2);\
};\
template <typename p0##_type, typename p1##_type>\
inline name##ActionP2<p0##_type, p1##_type> name(p0##_type p0, \
p1##_type p1) {\
return name##ActionP2<p0##_type, p1##_type>(p0, p1);\
}\
template <typename p0##_type, typename p1##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP2<p0##_type, p1##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P3(name, p0, p1, p2)\
template <typename p0##_type, typename p1##_type, typename p2##_type>\
class name##ActionP3 {\
public:\
name##ActionP3(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP3);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type>\
inline name##ActionP3<p0##_type, p1##_type, p2##_type> name(p0##_type p0, \
p1##_type p1, p2##_type p2) {\
return name##ActionP3<p0##_type, p1##_type, p2##_type>(p0, p1, p2);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP3<p0##_type, p1##_type, \
p2##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P4(name, p0, p1, p2, p3)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type>\
class name##ActionP4 {\
public:\
name##ActionP4(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3) : p0(gmock_p0), p1(gmock_p1), \
p2(gmock_p2), p3(gmock_p3) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP4);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type>\
inline name##ActionP4<p0##_type, p1##_type, p2##_type, \
p3##_type> name(p0##_type p0, p1##_type p1, p2##_type p2, \
p3##_type p3) {\
return name##ActionP4<p0##_type, p1##_type, p2##_type, p3##_type>(p0, p1, \
p2, p3);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP4<p0##_type, p1##_type, p2##_type, \
p3##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P5(name, p0, p1, p2, p3, p4)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type>\
class name##ActionP5 {\
public:\
name##ActionP5(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3, \
p4##_type gmock_p4) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4) : p0(gmock_p0), \
p1(gmock_p1), p2(gmock_p2), p3(gmock_p3), p4(gmock_p4) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3, p4));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP5);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type>\
inline name##ActionP5<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type> name(p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4) {\
return name##ActionP5<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type>(p0, p1, p2, p3, p4);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP5<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P6(name, p0, p1, p2, p3, p4, p5)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type>\
class name##ActionP6 {\
public:\
name##ActionP6(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3, p4, p5));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP6);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type>\
inline name##ActionP6<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type> name(p0##_type p0, p1##_type p1, p2##_type p2, \
p3##_type p3, p4##_type p4, p5##_type p5) {\
return name##ActionP6<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type>(p0, p1, p2, p3, p4, p5);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP6<p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P7(name, p0, p1, p2, p3, p4, p5, p6)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type>\
class name##ActionP7 {\
public:\
name##ActionP7(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5, p6##_type gmock_p6) : p0(gmock_p0), p1(gmock_p1), \
p2(gmock_p2), p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), \
p6(gmock_p6) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3, p4, p5, \
p6));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP7);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type>\
inline name##ActionP7<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type> name(p0##_type p0, p1##_type p1, \
p2##_type p2, p3##_type p3, p4##_type p4, p5##_type p5, \
p6##_type p6) {\
return name##ActionP7<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type>(p0, p1, p2, p3, p4, p5, p6);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP7<p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P8(name, p0, p1, p2, p3, p4, p5, p6, p7)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type>\
class name##ActionP8 {\
public:\
name##ActionP8(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5, p6##_type gmock_p6, \
p7##_type gmock_p7) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), \
p7(gmock_p7) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7) : p0(gmock_p0), \
p1(gmock_p1), p2(gmock_p2), p3(gmock_p3), p4(gmock_p4), \
p5(gmock_p5), p6(gmock_p6), p7(gmock_p7) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
p7##_type p7;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3, p4, p5, \
p6, p7));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
p7##_type p7;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP8);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type>\
inline name##ActionP8<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type, p7##_type> name(p0##_type p0, \
p1##_type p1, p2##_type p2, p3##_type p3, p4##_type p4, p5##_type p5, \
p6##_type p6, p7##_type p7) {\
return name##ActionP8<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type, p7##_type>(p0, p1, p2, p3, p4, p5, \
p6, p7);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP8<p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, \
p7##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P9(name, p0, p1, p2, p3, p4, p5, p6, p7, p8)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type>\
class name##ActionP9 {\
public:\
name##ActionP9(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5, p6##_type gmock_p6, p7##_type gmock_p7, \
p8##_type gmock_p8) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), p7(gmock_p7), \
p8(gmock_p8) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7, \
p8##_type gmock_p8) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), \
p7(gmock_p7), p8(gmock_p8) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
p7##_type p7;\
p8##_type p8;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3, p4, p5, \
p6, p7, p8));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
p7##_type p7;\
p8##_type p8;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP9);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type>\
inline name##ActionP9<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type, p7##_type, \
p8##_type> name(p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4, p5##_type p5, p6##_type p6, p7##_type p7, \
p8##_type p8) {\
return name##ActionP9<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type, p7##_type, p8##_type>(p0, p1, p2, \
p3, p4, p5, p6, p7, p8);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP9<p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type, \
p8##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

#define ACTION_P10(name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type, \
typename p9##_type>\
class name##ActionP10 {\
public:\
name##ActionP10(p0##_type gmock_p0, p1##_type gmock_p1, \
p2##_type gmock_p2, p3##_type gmock_p3, p4##_type gmock_p4, \
p5##_type gmock_p5, p6##_type gmock_p6, p7##_type gmock_p7, \
p8##_type gmock_p8, p9##_type gmock_p9) : p0(gmock_p0), p1(gmock_p1), \
p2(gmock_p2), p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), \
p7(gmock_p7), p8(gmock_p8), p9(gmock_p9) {}\
template <typename F>\
class gmock_Impl : public ::testing::ActionInterface<F> {\
public:\
typedef F function_type;\
typedef typename ::testing::internal::Function<F>::Result return_type;\
typedef typename ::testing::internal::Function<F>::ArgumentTuple\
args_type;\
gmock_Impl(p0##_type gmock_p0, p1##_type gmock_p1, p2##_type gmock_p2, \
p3##_type gmock_p3, p4##_type gmock_p4, p5##_type gmock_p5, \
p6##_type gmock_p6, p7##_type gmock_p7, p8##_type gmock_p8, \
p9##_type gmock_p9) : p0(gmock_p0), p1(gmock_p1), p2(gmock_p2), \
p3(gmock_p3), p4(gmock_p4), p5(gmock_p5), p6(gmock_p6), \
p7(gmock_p7), p8(gmock_p8), p9(gmock_p9) {}\
virtual return_type Perform(const args_type& args) {\
return ::testing::internal::ActionHelper<return_type, gmock_Impl>::\
Perform(this, args);\
}\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
return_type gmock_PerformImpl(const args_type& args, arg0_type arg0, \
arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4, \
arg5_type arg5, arg6_type arg6, arg7_type arg7, arg8_type arg8, \
arg9_type arg9) const;\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
p7##_type p7;\
p8##_type p8;\
p9##_type p9;\
private:\
GTEST_DISALLOW_ASSIGN_(gmock_Impl);\
};\
template <typename F> operator ::testing::Action<F>() const {\
return ::testing::Action<F>(new gmock_Impl<F>(p0, p1, p2, p3, p4, p5, \
p6, p7, p8, p9));\
}\
p0##_type p0;\
p1##_type p1;\
p2##_type p2;\
p3##_type p3;\
p4##_type p4;\
p5##_type p5;\
p6##_type p6;\
p7##_type p7;\
p8##_type p8;\
p9##_type p9;\
private:\
GTEST_DISALLOW_ASSIGN_(name##ActionP10);\
};\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type, \
typename p9##_type>\
inline name##ActionP10<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type, p7##_type, p8##_type, \
p9##_type> name(p0##_type p0, p1##_type p1, p2##_type p2, p3##_type p3, \
p4##_type p4, p5##_type p5, p6##_type p6, p7##_type p7, p8##_type p8, \
p9##_type p9) {\
return name##ActionP10<p0##_type, p1##_type, p2##_type, p3##_type, \
p4##_type, p5##_type, p6##_type, p7##_type, p8##_type, p9##_type>(p0, \
p1, p2, p3, p4, p5, p6, p7, p8, p9);\
}\
template <typename p0##_type, typename p1##_type, typename p2##_type, \
typename p3##_type, typename p4##_type, typename p5##_type, \
typename p6##_type, typename p7##_type, typename p8##_type, \
typename p9##_type>\
template <typename F>\
template <typename arg0_type, typename arg1_type, typename arg2_type, \
typename arg3_type, typename arg4_type, typename arg5_type, \
typename arg6_type, typename arg7_type, typename arg8_type, \
typename arg9_type>\
typename ::testing::internal::Function<F>::Result\
name##ActionP10<p0##_type, p1##_type, p2##_type, p3##_type, p4##_type, \
p5##_type, p6##_type, p7##_type, p8##_type, \
p9##_type>::gmock_Impl<F>::gmock_PerformImpl(\
GMOCK_ACTION_ARG_TYPES_AND_NAMES_UNUSED_) const

namespace testing {


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif


namespace internal {
namespace invoke_argument {

struct AdlTag {};


template <typename R, typename F>
R InvokeArgumentAdl(AdlTag, F f) {
return f();
}
template <typename R, typename F, typename A1>
R InvokeArgumentAdl(AdlTag, F f, A1 a1) {
return f(a1);
}
template <typename R, typename F, typename A1, typename A2>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2) {
return f(a1, a2);
}
template <typename R, typename F, typename A1, typename A2, typename A3>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3) {
return f(a1, a2, a3);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4) {
return f(a1, a2, a3, a4);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4, typename A5>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5) {
return f(a1, a2, a3, a4, a5);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4, typename A5, typename A6>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6) {
return f(a1, a2, a3, a4, a5, a6);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4, typename A5, typename A6, typename A7>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6,
A7 a7) {
return f(a1, a2, a3, a4, a5, a6, a7);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4, typename A5, typename A6, typename A7, typename A8>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6,
A7 a7, A8 a8) {
return f(a1, a2, a3, a4, a5, a6, a7, a8);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4, typename A5, typename A6, typename A7, typename A8,
typename A9>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6,
A7 a7, A8 a8, A9 a9) {
return f(a1, a2, a3, a4, a5, a6, a7, a8, a9);
}
template <typename R, typename F, typename A1, typename A2, typename A3,
typename A4, typename A5, typename A6, typename A7, typename A8,
typename A9, typename A10>
R InvokeArgumentAdl(AdlTag, F f, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6,
A7 a7, A8 a8, A9 a9, A10 a10) {
return f(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
}
}  
}  

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args));
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(p0)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_2_VALUE_PARAMS(p0, p1)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_3_VALUE_PARAMS(p0, p1, p2)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_4_VALUE_PARAMS(p0, p1, p2, p3)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3, p4);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3, p4, p5);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3, p4, p5, p6);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3, p4, p5, p6, p7);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7, p8)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3, p4, p5, p6, p7, p8);
}

ACTION_TEMPLATE(InvokeArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)) {
using internal::invoke_argument::InvokeArgumentAdl;
return InvokeArgumentAdl<return_type>(
internal::invoke_argument::AdlTag(),
::testing::get<k>(args), p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_0_VALUE_PARAMS()) {
return new T();
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_1_VALUE_PARAMS(p0)) {
return new T(p0);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_2_VALUE_PARAMS(p0, p1)) {
return new T(p0, p1);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_3_VALUE_PARAMS(p0, p1, p2)) {
return new T(p0, p1, p2);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_4_VALUE_PARAMS(p0, p1, p2, p3)) {
return new T(p0, p1, p2, p3);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_5_VALUE_PARAMS(p0, p1, p2, p3, p4)) {
return new T(p0, p1, p2, p3, p4);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_6_VALUE_PARAMS(p0, p1, p2, p3, p4, p5)) {
return new T(p0, p1, p2, p3, p4, p5);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_7_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6)) {
return new T(p0, p1, p2, p3, p4, p5, p6);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_8_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7)) {
return new T(p0, p1, p2, p3, p4, p5, p6, p7);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_9_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7, p8)) {
return new T(p0, p1, p2, p3, p4, p5, p6, p7, p8);
}

ACTION_TEMPLATE(ReturnNew,
HAS_1_TEMPLATE_PARAMS(typename, T),
AND_10_VALUE_PARAMS(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)) {
return new T(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

}  

#include "gmock/internal/custom/gmock-generated-actions.h"

#endif  
