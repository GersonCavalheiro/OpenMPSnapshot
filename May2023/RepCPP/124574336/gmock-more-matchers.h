



#ifndef GMOCK_INCLUDE_GMOCK_MORE_MATCHERS_H_
#define GMOCK_INCLUDE_GMOCK_MORE_MATCHERS_H_

#include "gmock/gmock-matchers.h"

namespace testing {

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4100)
#if (_MSC_VER == 1900)
# pragma warning(disable:4800)
#endif
#endif

MATCHER(IsEmpty, negation ? "isn't empty" : "is empty") {
if (arg.empty()) {
return true;
}
*result_listener << "whose size is " << arg.size();
return false;
}

MATCHER(IsTrue, negation ? "is false" : "is true") {
return static_cast<bool>(arg);
}

MATCHER(IsFalse, negation ? "is true" : "is false") {
return !static_cast<bool>(arg);
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif


}  

#endif  
