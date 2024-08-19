#include <stdint.h>
extern uint32_t bar ();
#pragma GCC target("fpu=vfpv3-d16")
extern float fmaf (float, float, float);
float
__attribute__((target("fpu=vfpv4"))) vfma32 (float x, float y, float z)
{
return fmaf (x, y, z);
}
uint32_t restored ()
{
return bar();
}
