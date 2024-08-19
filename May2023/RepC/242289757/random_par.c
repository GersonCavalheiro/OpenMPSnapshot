
#include <omp.h>

static unsigned long long MULTIPLIER  = 764261123;
static unsigned long long PMOD        = 2147483647;
static unsigned long long mult_n;
double random_low, random_hi;

#define MAX_THREADS 128
static unsigned long long pseed[MAX_THREADS][4]; 
unsigned long long random_last = 0;
#pragma omp threadprivate(random_last)


double drandom()
{
unsigned long long random_next;
double ret_val;

random_next = (unsigned long long)((mult_n  * random_last)% PMOD);
random_last = random_next;

ret_val = ((double)random_next/(double)PMOD)*(random_hi-random_low)+random_low;
return ret_val;
}

void seed(double low_in, double hi_in)
{
int i, id, nthreads;
unsigned long long iseed;
id = omp_get_thread_num();

#pragma omp single
{
if(low_in < hi_in)
{ 
random_low = low_in;
random_hi  = hi_in;
}
else
{
random_low = hi_in;
random_hi  = low_in;
}


nthreads = omp_get_num_threads();
iseed = PMOD/MULTIPLIER;     
pseed[0][0] = iseed;
mult_n = MULTIPLIER;
for (i = 1; i < nthreads; ++i)
{
iseed = (unsigned long long)((MULTIPLIER * iseed) % PMOD);
pseed[i][0] = iseed;
mult_n = (mult_n * MULTIPLIER) % PMOD;
}

}
random_last = (unsigned long long) pseed[id][0];
}

