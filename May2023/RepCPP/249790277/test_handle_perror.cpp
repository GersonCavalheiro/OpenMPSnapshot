


#include <cerrno>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <stdexcept>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif

#include "../tbb/tbb_misc.h"
#include "harness.h"

#if TBB_USE_EXCEPTIONS

static void TestHandlePerror() {
bool caught = false;
try {
tbb::internal::handle_perror( EAGAIN, "apple" );
} catch( std::runtime_error& e ) {
#if TBB_USE_EXCEPTIONS
REMARK("caught runtime_exception('%s')\n",e.what());
ASSERT( memcmp(e.what(),"apple: ",7)==0, NULL );
ASSERT( strlen(strstr(e.what(), strerror(EAGAIN))), "bad error message?" );
#endif 
caught = true;
}
ASSERT( caught, NULL );
}

int TestMain () {
TestHandlePerror();
return Harness::Done;
}

#else 

int TestMain () {
return Harness::Skipped;
}

#endif 
