#include <vecintrin.h>
__attribute__ ((target("arch=z13")))
void vx1(void)
{
vec_load_bndry ((const signed char *)0, 64);
__builtin_s390_vll ((unsigned int)0, (const void *)8);
}
__attribute__ ((target("arch=z10")))
void vx0(void)
{
__builtin_s390_vll ((unsigned int)0, (const void *)8); 
}
void vxd(void)
{
vec_load_bndry ((const signed char *)0, 64);
__builtin_s390_vll ((unsigned int)0, (const void *)8);
}
