

#include "rawspeedconfig.h" 
#include "common/Cpuid.h"   
#include <cstdlib>          
#include <string>           
#include <gtest/gtest.h>    

using rawspeed::Cpuid;

namespace rawspeed_test {

TEST(CpuidDeathTest, SSE2Test) {
#if defined(__SSE2__)
ASSERT_EXIT(
{
ASSERT_TRUE(Cpuid::SSE2());
exit(0);
},
::testing::ExitedWithCode(0), "");
#else
ASSERT_EXIT(
{
ASSERT_FALSE(Cpuid::SSE2());
exit(0);
},
::testing::ExitedWithCode(0), "");
#endif
}

} 
