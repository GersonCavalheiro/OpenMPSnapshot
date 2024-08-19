



#ifndef GMOCK_INCLUDE_GMOCK_GMOCK_MORE_ACTIONS_H_
#define GMOCK_INCLUDE_GMOCK_GMOCK_MORE_ACTIONS_H_

#include <algorithm>
#include <type_traits>

#include "gmock/gmock-generated-actions.h"

namespace testing {
namespace internal {

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


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#endif

ACTION_TEMPLATE(ReturnArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
return ::std::get<k>(args);
}

ACTION_TEMPLATE(SaveArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(pointer)) {
*pointer = ::std::get<k>(args);
}

ACTION_TEMPLATE(SaveArgPointee,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(pointer)) {
*pointer = *::std::get<k>(args);
}

ACTION_TEMPLATE(SetArgReferee,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_1_VALUE_PARAMS(value)) {
typedef typename ::std::tuple_element<k, args_type>::type argk_type;
GTEST_COMPILE_ASSERT_(std::is_reference<argk_type>::value,
SetArgReferee_must_be_used_with_a_reference_argument);
::std::get<k>(args) = value;
}

ACTION_TEMPLATE(SetArrayArgument,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_2_VALUE_PARAMS(first, last)) {
#ifdef _MSC_VER
internal::CopyElements(first, last, ::std::get<k>(args));
#else
::std::copy(first, last, ::std::get<k>(args));
#endif
}

ACTION_TEMPLATE(DeleteArg,
HAS_1_TEMPLATE_PARAMS(int, k),
AND_0_VALUE_PARAMS()) {
delete ::std::get<k>(args);
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
