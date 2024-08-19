#ifndef GENETICALGORITHM_LIGHTRAND_H
#define GENETICALGORITHM_LIGHTRAND_H
#include <stdlib.h>
#pragma offload_attribute(push, target(mic))
class LightRand {
public:
static unsigned long pa, npa, seed;
static unsigned long x, y, z;
static unsigned int Rand();
};
#pragma offload_attribute(pop)
#endif 
