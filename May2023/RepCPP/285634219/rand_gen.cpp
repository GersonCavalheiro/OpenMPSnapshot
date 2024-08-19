
#include "randomc.h"

#define LOWER_MASK ((1LU << MERS_R) - 1)         
#define UPPER_MASK (0xFFFFFFFF << MERS_R)        


void RandomInit(uint32 *mt, uint32 seed) 
{
int i;
if(omp_get_thread_num() == 0)
{
mt[0]= seed & 0xffffffffUL;
for (i=1; i < MERS_N; i++) 
{
mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
}
}
#pragma omp barrier
}

void BRandom(uint32* mt) 
{
uint32 y;
int thdx;
int threadIdx_x = omp_get_thread_num();

if (threadIdx_x<MERS_N-MERS_M) 
{
y = (mt[threadIdx_x] & UPPER_MASK) | (mt[threadIdx_x+1] & LOWER_MASK);
y = mt[threadIdx_x+MERS_M] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
}
#pragma omp barrier
if (threadIdx_x<MERS_N-MERS_M) 
{
mt[threadIdx_x] = y;
}
#pragma omp barrier

thdx = threadIdx_x + (MERS_N-MERS_M);
if (threadIdx_x<MERS_N-MERS_M) 
{
y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
y = mt[threadIdx_x] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
}
#pragma omp barrier
if (threadIdx_x<MERS_N-MERS_M) 
{
mt[thdx] = y;
}
#pragma omp barrier

thdx += (MERS_N-MERS_M);
if (thdx < MERS_N-1) 
{
y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
y = mt[threadIdx_x+(MERS_N-MERS_M)] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
}
#pragma omp barrier
if (thdx < MERS_N-1) 
{
mt[thdx] = y;
}
#pragma omp barrier

if (threadIdx_x == 0) 
{
y = (mt[MERS_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
mt[MERS_N-1] = mt[MERS_M-1] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
}
#pragma omp barrier

y ^=  y >> MERS_U;
y ^= (y << MERS_S) & MERS_B;
y ^= (y << MERS_T) & MERS_C;
y ^=  y >> MERS_L;

}



