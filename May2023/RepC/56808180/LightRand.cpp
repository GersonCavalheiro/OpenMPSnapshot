#include "LightRand.h"
#pragma offload_attribute(push, target(mic))
unsigned long LightRand::x=123456789, LightRand::y=362436069, LightRand::z=5211288629, LightRand::seed = 0, LightRand::pa = 0, LightRand::npa = 0;
unsigned int LightRand::Rand() {
unsigned long t;
x ^= x << 16;
x ^= x << 5;
x ^= x << 1;
t = x;
x = y;
y = z;
z = t ^ x ^ y;
if((z + seed)%2){
pa++;
}
else
npa++;
return rand();
}
#pragma offload_attribute(pop)