#ifndef _KERNEL_PRNG_SETUP_
#define _KERNEL_PRNG_SETUP_






#include <limits.h>
#include <inttypes.h>

#define INV_UINT_MAX 2.3283064e-10f

#pragma omp declare target
inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc);

inline void gpu_pcg32_srandom_r(uint64_t *state, uint64_t *inc, uint64_t initstate, uint64_t initseq)
{
*state = 0U;
*inc = (initseq << 1u) | 1u;
gpu_pcg32_random_r(state, inc);
*state += initstate;
gpu_pcg32_random_r(state, inc);
}


inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc)
{
uint64_t oldstate = *state;
*state = oldstate * 6364136223846793005ULL + *inc;
uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
uint32_t rot = oldstate >> 59u;
return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

inline float gpu_rand01(uint64_t *state, uint64_t *inc)
{
return (float) gpu_pcg32_random_r(state, inc) * INV_UINT_MAX;
}



uint64_t mmhash64( const void * key, int len, unsigned int seed )
{
const uint64_t m = 0xc6a4a7935bd1e995;
const int r = 47;

uint64_t h = seed ^ (len * m);

const uint64_t * data = (const uint64_t *)key;
const uint64_t * end = data + (len/8);

while(data != end){
uint64_t k = *data++;

k *= m; 
k ^= k >> r; 
k *= m; 

h ^= k;
h *= m; 
}
const unsigned char * data2 = (const unsigned char*)data;
switch(len & 7)
{
case 7: h ^= uint64_t(data2[6]) << 48;
case 6: h ^= uint64_t(data2[5]) << 40;
case 5: h ^= uint64_t(data2[4]) << 32;
case 4: h ^= uint64_t(data2[3]) << 24;
case 3: h ^= uint64_t(data2[2]) << 16;
case 2: h ^= uint64_t(data2[1]) << 8;
case 1: h ^= uint64_t(data2[0]);
h *= m;
};

h ^= h >> r;
h *= m;
h ^= h >> r;
return h;
} 
#pragma omp end declare target

void kernel_gpupcg_setup(uint64_t *state, uint64_t *inc, int N, 
uint64_t seed, uint64_t seq)
{
#pragma omp target teams distribute parallel for thread_limit(BLOCKSIZE1D)
for (int x = 0; x < N; x++) {
uint64_t tseed = x + seed;
uint64_t hseed = mmhash64(&tseed, sizeof(uint64_t), 17);
uint64_t hseq = mmhash64(&seq, sizeof(uint64_t), 47);
gpu_pcg32_srandom_r(&state[x], &inc[x], hseed, hseq);
}
}
#endif

