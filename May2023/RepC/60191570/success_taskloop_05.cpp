#include<assert.h>
#include<stdio.h>
#define N 1000
#define NUM_TASKS 20
void strictly_increasing_loop(int l, int u, int s, int ntasks)
{
int var = 0;
char first_time = 1;
#pragma omp taskloop num_tasks(ntasks) firstprivate(first_time) shared(var)
for (int j = l; j < u; j += s)
{
if (first_time)
{
#pragma omp atomic
var++;
first_time = 0;
}
}
assert(var == ntasks);
}
void strictly_decreasing_loop(int l, int u, int s, int ntasks)
{
int var = 0;
char first_time = 1;
#pragma omp taskloop num_tasks(ntasks) firstprivate(first_time) shared(var)
for (int j = l; j >= u; j += s)
{
if (first_time)
{
#pragma omp atomic
var++;
first_time = 0;
}
}
assert(var == ntasks);
}
int main(int argc, char* argv[])
{
int var;
for(int ntasks = 1; ntasks <= 20; ++ntasks)
{
strictly_increasing_loop(0, 100,  1, ntasks);
strictly_decreasing_loop(99,  0, -1, ntasks);
strictly_increasing_loop(-100,  0,  1, ntasks);
strictly_decreasing_loop(-1, -100, -1, ntasks);
strictly_increasing_loop(-100, 100,  1, ntasks);
strictly_decreasing_loop(99,  -100, -1, ntasks);
}
return 0;
}
