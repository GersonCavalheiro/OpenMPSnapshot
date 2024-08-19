



#ifndef harness_preload_H
#define harness_preload_H

#if __GNUC__>=5 && !__INTEL_COMPILER && !__clang__ && __GXX_EXPERIMENTAL_CXX0X__
#pragma GCC diagnostic warning "-Wsuggest-override"
#define __TBB_TEST_USE_WSUGGEST_OVERRIDE 1
#endif

#if __TBB_TEST_NO_EXCEPTIONS
#include "tbb/tbb_disable_exceptions.h"
#endif

#endif 
