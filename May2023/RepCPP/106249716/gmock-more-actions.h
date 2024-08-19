

#ifndef GMOCK_INCLUDE_GMOCK_GMOCK_MORE_ACTIONS_H_
#define GMOCK_INCLUDE_GMOCK_GMOCK_MORE_ACTIONS_H_

#include <algorithm>

#include "gmock/gmock-generated-actions.h"

namespace testing {
namespace internal {

template <typename FunctionImpl>
class InvokeAction {
public:
explicit InvokeAction(FunctionImpl function_impl)
: function_impl_(function_impl) {}

template <typename Result, typename ArgumentTuple>
Result Perform(const ArgumentTuple& args) {
return InvokeHelper<Result, ArgumentTuple>::Invoke(function_impl_, args);
}

private:
FunctionImpl function_impl_;

GTEST_DISALLOW_ASSIGN_(InvokeAction);
};

template <class Class, typename MethodPtr>
class InvokeMethodAction {
public:
InvokeMethodAction(Class* obj_ptr, MethodPtr method_ptr)
: method_ptr_(method_ptr), obj_ptr_(obj_ptr) {}

template <typename Result, typename ArgumentTuple>
Result Perform(const ArgumentTuple& args) const {
return InvokeHelper<Result, ArgumentTuple>::InvokeMethod(
obj_ptr_, method_ptr_, args);
}

private:
const MethodPtr method_ptr_;
Class* const obj_ptr_;

GTEST_DISALLOW_ASSIGN_(InvokeMethodAction);
};

template<typename InputIterator, typename OutputIterator>
inline OutputIterator CopyElements(InputIterator first,
InputIterator last,
OutputIterator output) {
for (; first != last; ++first, ++output) {
*output = *first;
}
return output;
}

}  


template <typename FunctionImpl>
PolymorphicAction<internal::InvokeAction<FunctionImpl> > Invoke(
FunctionImpl function_impl) {
return MakePolymorphicAction(
internal::InvokeAction<FunctionImpl>(function_impl));
}

template <class Class, typename MethodPtr>
PolymorphicAction<internal::InvokeMethodAction<Class, MethodPtr> > Invoke(
Class* obj_ptr, MethodPtr method_ptr) {
return MakePolymorphicAction(
internal::InvokeMethodAction<Class, MethodPtr>(obj_ptr, method_ptr));
}

template <typename InnerAction>
inline internal::WithArgsAction<InnerAction>
WithoutArgs(const InnerAction& action) {
return internal::WithArgsAction<InnerAction>(action);
}

template <int k, typename InnerAction>
inline internal::WithArgsAction<InnerAction, k>
WithArg(const InnerAction& action) {
return internal::WithArgsAction<InnerAction, k>(action);
}

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif

ACTION_TEMPLATE(ReturnArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
return ::testing::get<k>(args);
}

ACTION_TEMPLATE(SaveArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(pointer)) {
*pointer = ::testing::get<k>(args);
}

ACTION_TEMPLATE(SaveArgPointee,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(pointer)) {
*pointer = *::testing::get<k>(args);
}

ACTION_TEMPLATE(SetArgReferee,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(value)) {
typedef typename ::testing::tuple_element<k, args_type>::type argk_type;
GTEST_COMPILE_ASSERT_(internal::is_reference<argk_type>::value,
SetArgReferee_must_be_used_with_a_reference_argument);
::testing::get<k>(args) = value;
}

ACTION_TEMPLATE(SetArrayArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_2_VALUE_PARAMS(first, last)) {
#ifdef _MSC_VER
internal::CopyElements(first, last, ::testing::get<k>(args));
#else
::std::copy(first, last, ::testing::get<k>(args));
#endif
}

ACTION_TEMPLATE(DeleteArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
delete ::testing::get<k>(args);
}

ACTION_P(ReturnPointee, pointer) { return *pointer; }

#if GTEST_HAS_EXCEPTIONS

# ifdef _MSC_VER
#  pragma warning(push)          
#  pragma warning(disable:4702)  
# endif
ACTION_P(Throw, exception) { throw exception; }
# ifdef _MSC_VER
#  pragma warning(pop)           
# endif

#endif  

#ifdef _MSC_VER
# pragma warning(pop)
#endif

}  

#endif  
