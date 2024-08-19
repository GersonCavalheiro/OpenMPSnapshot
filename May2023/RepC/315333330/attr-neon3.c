#pragma GCC target ("fpu=vfp")
#pragma GCC push_options
#include <arm_neon.h>
int8x8_t __attribute__ ((target("fpu=neon")))
my (int8x8_t __a, int8x8_t __b)
{
return __a + __b;
}
poly128_t __attribute__ ((target("fpu=crypto-neon-fp-armv8")))
foo (poly128_t* ptr)
{
return vldrq_p128 (ptr);
}
int8x8_t
my1 (int8x8_t __a, int8x8_t __b)
{
return __a + __b;
}
