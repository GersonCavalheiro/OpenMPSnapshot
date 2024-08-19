#include<stdio.h>
#include<string.h>
#include<assert.h>
void validate(int *v, int start, int end, int val)
{
for (int i = start; i < end; ++i)
assert(v[i] == val);
}
#define N 50
#define BEG  0
#define MID 25
#define END N
#define INIT \
memset(v, 0, N*sizeof(int));
#define VALIDATE \
validate(v, BEG, MID,  1); \
validate(v, MID, END,  0);
int main()
{
int v[N];
for (int k = 1; k <= N; ++k) {
INIT
#pragma omp taskloop shared(v) grainsize(k)
for (int i = BEG; i < MID; ++i)
v[i]++;
VALIDATE
INIT
#pragma omp taskloop shared(v) grainsize(k)
for (int i = BEG; i <= MID-1; ++i)
v[i]++;
VALIDATE
INIT
#pragma omp taskloop shared(v) grainsize(k)
for (int i = MID-1; i > BEG-1; --i)
v[i]++;
VALIDATE
INIT
#pragma omp taskloop shared(v) grainsize(k)
for (int i = MID-1; i >= BEG; --i)
v[i]++;
VALIDATE
}
for (int k = 1; k <= N; ++k) {
INIT
#pragma omp taskloop shared(v) num_tasks(k)
for (int i = BEG; i < MID; ++i)
v[i]++;
VALIDATE
INIT
#pragma omp taskloop shared(v) num_tasks(k)
for (int i = BEG; i <= MID-1; ++i)
v[i]++;
VALIDATE
INIT
#pragma omp taskloop shared(v) num_tasks(k)
for (int i = MID-1; i > BEG-1; --i)
v[i]++;
VALIDATE
INIT
#pragma omp taskloop shared(v) num_tasks(k)
for (int i = MID-1; i >= BEG; --i)
v[i]++;
VALIDATE
}
}
