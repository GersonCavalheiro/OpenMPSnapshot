#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#define MOD 10000007
#define MULTIPLIER 1234
#define INCREMENT 12345
long long seed;
#pragma omp threadprivate(seed)
long long leap_multiplier = MULTIPLIER, leap_increment = INCREMENT;
long lcg()
{
seed = (leap_multiplier*seed + leap_increment)%MOD;
return seed;
}
void set_pseudo_rand_seed(long p_seed)
{
seed = p_seed;
}
double pseudo_rand()
{
long rand_num = lcg();
return 2.0*rand_num/(MOD-1);
}
long long modexp(long long base, long long exp)
{
long long ans = 1;
while(exp)
{
if(exp % 2)
ans = (ans * base) % MOD;
base = (base * base) % MOD;
exp = exp >> 1; 
}
return ans;
}
double seq_pimonte(long num_steps,int r)
{
double cx =1.0, cy = 1.0;
double px,py,d;
double count = 0;
set_pseudo_rand_seed(1234);
for (int i = 0; i < num_steps; ++i)
{
px = pseudo_rand();
py = pseudo_rand();
d = sqrt((px - 1)*(px - 1) + (py - 1)*(py - 1));
if(d <= r)
{
count++;
}
}
return 4.0*count/num_steps;
}
double parallel_pimonte(long num_steps,int r,int NUM_THREADS)
{
double cx =1.0, cy = 1.0;
double px,py,d,count=0;
int nthreads,t_seeds[20];
omp_set_num_threads(NUM_THREADS);
#pragma omp parallel private(px,py,d)
{
#pragma omp single
{
nthreads = omp_get_num_threads();
t_seeds[0] = MOD/MULTIPLIER;
for(int i=1;i<nthreads;++i)
t_seeds[i] = ((MULTIPLIER*t_seeds[i-1] + MOD)%MOD + INCREMENT + MOD)%MOD;
leap_multiplier = modexp(MULTIPLIER,nthreads);
leap_increment = ((INCREMENT*(leap_multiplier - 1 + MOD)%MOD)*modexp(MULTIPLIER - 1,MOD - 2)+MOD)%MOD;
}
int id = omp_get_thread_num();
set_pseudo_rand_seed(t_seeds[id]);
#pragma omp for reduction(+:count)
for(int i=0;i<num_steps;++i)
{
px = pseudo_rand();
py = pseudo_rand();
d = sqrt((px - 1)*(px - 1) + (py - 1)*(py - 1));
if(d <= r)
{
count = count + 1;
}
}
}
return 4.0*count/num_steps;
}
int main()
{
long num_steps = 10000000;
double time_taken_seq,time_taken_parallel;
double pi;
time_taken_seq = omp_get_wtime();
pi = seq_pimonte(num_steps,1);
time_taken_seq = omp_get_wtime() - time_taken_seq;
printf("Sequential program Pi : %lf \n", pi);
printf("Parallel Calculation\n");
int NUM_THREADS = 2;
while(NUM_THREADS<=20)
{
time_taken_parallel = omp_get_wtime();
pi = parallel_pimonte(num_steps,1,NUM_THREADS);
time_taken_parallel = omp_get_wtime() - time_taken_parallel;
printf("Pi :  %lf \t Speedup: %lf \t Threads : %d\n", pi,time_taken_parallel/time_taken_seq,NUM_THREADS);		
NUM_THREADS++;
}
}