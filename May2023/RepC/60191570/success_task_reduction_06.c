#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<assert.h>
#define N 1000
int main()
{
int v[N];
int serial_max_val = 0, max_val = 0;
int i;
srand(time(NULL));
for(i = 0; i < N; i++)
{
v[i] = rand() % N;
if (serial_max_val < v[i])
{
serial_max_val = v[i];
}
}
for(i = 0;i < N; i++)
{
#pragma omp task reduction(max : max_val) firstprivate(i) shared(v)
{
if(v[i] > max_val)
{
max_val = v[i];
}
}
}
#pragma omp task in(max_val)
{
assert(serial_max_val == max_val);
}
#pragma omp taskwait
}
