
#ifndef _PETRINET_KERNEL_H_
#define _PETRINET_KERNEL_H_

#include "petri.h"

#define BLOCK_SIZE 256
#define BLOCK_SIZE_BITS 8

#pragma omp declare target
void fire_transition(char* g_places, int* conflict_array, int tr, 
int tc, int step, int N, int thd_thrd) 
{
int val1, val2, val3, to_update;
int mark1, mark2;

to_update = 0;
if (omp_get_thread_num() < thd_thrd) 
{
val1 = (tr==0)? (N+N)-1: tr-1;
val2 = (tr & 0x1)? (tc==N-1? 0: tc+1): tc;
val3 = (tr==(N+N)-1)? 0: tr+1;
mark1 = g_places[val1*N+val2];
mark2 = g_places[tr*N+tc];
if ( (mark1>0) && (mark2>0) ) 
{
to_update = 1;
conflict_array[tr*N+tc] = step;
}
}
#pragma omp barrier

if (to_update) 
{
to_update = ((step & 0x01) == (tr & 0x01) ) || 
( (conflict_array[val1*N+val2]!=step) && 
(conflict_array[val3*N+((val2==0)? N-1: val2-1)]!=step) );
}

if (to_update) 
{
g_places[val1*N+val2] = mark1-1;  
g_places[tr*N+tc] = mark2-1; 
}
#pragma omp barrier
if (to_update) 
{
g_places[val3*N+val2]++;  
g_places[tr*N+(tc==N-1? 0: tc+1)]++; 
}
#pragma omp barrier
}


void initialize_grid(uint32* mt, int* g_places, int nsquare2, int seed) 
{
int i;
int loop_num = nsquare2 >> (BLOCK_SIZE_BITS+2);
int threadIdx_x = omp_get_thread_num();

for (i=0; i<loop_num; i++) 
{
g_places[threadIdx_x+(i<<BLOCK_SIZE_BITS)] = 0x01010101;
}

if (threadIdx_x < (nsquare2>>2)-(loop_num<<BLOCK_SIZE_BITS)) 
{
g_places[threadIdx_x+(loop_num<<BLOCK_SIZE_BITS)] = 0x01010101;
}

RandomInit(mt, omp_get_team_num() + seed);
}

void run_trajectory(uint32* mt,
int* g_places, int n, int max_steps) 
{
int step, nsquare2, val;

step = 0;
nsquare2 = (n+n)*n;

int threadIdx_x = omp_get_thread_num();

while (step<max_steps) 
{
BRandom(mt); 

val = mt[threadIdx_x]%nsquare2;
fire_transition((char*)g_places, g_places+(nsquare2>>2), 
val/n, val%n, step+7, n, BLOCK_SIZE);

val = mt[threadIdx_x+BLOCK_SIZE]%nsquare2;
fire_transition((char*)g_places, g_places+(nsquare2>>2), 
val/n, val%n, step+11, n, BLOCK_SIZE);

if (  threadIdx_x < MERS_N-(BLOCK_SIZE<<1)  ) 
{
val = mt[threadIdx_x+(BLOCK_SIZE<<1)]%nsquare2;
}
fire_transition((char*)g_places, g_places+(nsquare2>>2), 
val/n, val%n, step+13, n, MERS_N-(BLOCK_SIZE<<1));

step += MERS_N>>1; 
}
}


void compute_reward_stat(
uint32* __restrict mt,
int* __restrict g_places,
float* __restrict g_vars,
int* __restrict g_maxs, 
int nsquare2) 
{
float sum = 0;
int i;
int max = 0;
int temp, data; 
int loop_num = nsquare2 >> (BLOCK_SIZE_BITS+2);
int threadIdx_x = omp_get_thread_num();
int blockIdx_x = omp_get_team_num();

for (i=0; i<=loop_num-1; i++) 
{ 
data = g_places[threadIdx_x+(i<<BLOCK_SIZE_BITS)];

temp = data & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
temp = (data>>8) & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
temp = (data>>16) & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
temp = (data>>24) & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
}

i = nsquare2>>2;
i &= 0x0FF;
loop_num *= BLOCK_SIZE; 
if (threadIdx_x <= i-1) 
{
data = g_places[threadIdx_x+loop_num];

temp = data & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
temp = (data>>8) & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
temp = (data>>16) & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
temp = (data>>24) & 0x0FF;
sum += temp*temp;
max = max<temp? temp: max;
}

((float*)mt)[threadIdx_x] = (float)sum;
mt[threadIdx_x+BLOCK_SIZE] = (uint32)max;
#pragma omp barrier

for (i=(BLOCK_SIZE>>1); i>0; i = (i>>1) ) 
{
if (threadIdx_x<i) 
{
((float*)mt)[threadIdx_x] += ((float*)mt)[threadIdx_x+i];
if (mt[threadIdx_x+BLOCK_SIZE]<mt[threadIdx_x+i+BLOCK_SIZE])
mt[threadIdx_x+BLOCK_SIZE] = mt[threadIdx_x+i+BLOCK_SIZE];
}
#pragma omp barrier
}

if (threadIdx_x==0) 
{
g_vars[blockIdx_x] = (((float*)mt)[0])/nsquare2-1; 
g_maxs[blockIdx_x] = (int)mt[BLOCK_SIZE];
}
}
#pragma omp end declare target

void PetrinetKernel(
uint32* __restrict mt,
int* __restrict g_s,
float* __restrict g_v,
int* __restrict g_m,
int n, int s, int seed)
{
int nsquare2 = n*n*2;
int* g_places = g_s + omp_get_team_num() * ((nsquare2>>2)+nsquare2);   
initialize_grid(mt, g_places, nsquare2, seed);

run_trajectory(mt, g_places, n, s);
compute_reward_stat(mt, g_places, g_v, g_m, nsquare2);
}

#endif 
