

#pragma once

#include <cstdio>

#ifdef ALPAKA_ACC_SYCL_ENABLED
#    define ALPAKA_CHECK(success, expression)                                                                         \
do                                                                                                            \
{                                                                                                             \
if(!(expression))                                                                                         \
{                                                                                                         \
acc.cout << "ALPAKA_CHECK failed because '!(" << #expression << ")'\n";                               \
success = false;                                                                                      \
}                                                                                                         \
} while(0)
#else
#    define ALPAKA_CHECK(success, expression)                                                                         \
do                                                                                                            \
{                                                                                                             \
if(!(expression))                                                                                         \
{                                                                                                         \
printf("ALPAKA_CHECK failed because '!(%s)'\n", #expression);                                         \
success = false;                                                                                      \
}                                                                                                         \
} while(0)
#endif
