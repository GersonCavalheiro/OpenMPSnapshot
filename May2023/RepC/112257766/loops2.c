#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "workqueue.h"
#define N 729
#define reps 1000
double a[N][N], b[N][N], c[N];
int jmax[N];
void init1(void);
void init2(void);
void runloop(int);
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);
void runloopchunk(int, int, int);
int main(int argc, char *argv[])
{
double start1, start2, end1, end2;
int r;
init1();
start1 = omp_get_wtime();
for (r = 0; r < reps; r++)
{
runloop(1);
}
end1 = omp_get_wtime();
valid1();
printf("Total time for %d reps of loop 1 = %f\n", reps, (float)(end1 - start1));
init2();
start2 = omp_get_wtime();
for (r = 0; r < reps; r++)
{
runloop(2);
}
end2 = omp_get_wtime();
valid2();
printf("Total time for %d reps of loop 2 = %f\n", reps, (float)(end2 - start2));
}
void init1(void)
{
int i, j;
for (i = 0; i < N; i++)
{
for (j = 0; j < N; j++)
{
a[i][j] = 0.0;
b[i][j] = 3.142 * (i + j);
}
}
}
void init2(void)
{
int i, j, expr;
for (i = 0; i < N; i++)
{
expr = i % (3 * (i / 30) + 1);
if (expr == 0)
{
jmax[i] = N;
}
else
{
jmax[i] = 1;
}
c[i] = 0.0;
}
for (i = 0; i < N; i++)
{
for (j = 0; j < N; j++)
{
b[i][j] = (double)(i * j + 1) / (double)(N * N);
}
}
}
void runloop(int loopid)
{
work_queue *work_queues;      
omp_lock_t *work_queue_locks; 
#pragma omp parallel default(none) shared(loopid, work_queues, work_queue_locks)
{
int myid = omp_get_thread_num();
int nthreads = omp_get_num_threads();
#pragma omp single
{
work_queues = (work_queue *)malloc(nthreads * sizeof(work_queue));
work_queue_locks = (omp_lock_t *)malloc(nthreads * sizeof(omp_lock_t));
for (int i = 0; i < nthreads; ++i)
{
init_work_queue(&work_queues[i]);
omp_init_lock(&work_queue_locks[i]);
}
}
int ipt = (int)ceil((double)N / (double)nthreads);
int local_work_set_start = myid * ipt;
int local_work_set_end = (myid + 1) * ipt;
local_work_set_end = (local_work_set_end > N) ? N : local_work_set_end;
while (local_work_set_start < local_work_set_end)
{
int workload = (int)ceil((double)(local_work_set_end - local_work_set_start) / (double)nthreads);
chunk new_chunk;
new_chunk.lo = local_work_set_start;
new_chunk.hi = local_work_set_start + workload;
new_chunk.workload = workload;
enworkqueue(&work_queues[myid], new_chunk);
local_work_set_start = local_work_set_start + workload;
}
#pragma omp barrier
while (!is_queue_empty(&work_queues[myid]))
{
chunk current_work;
omp_set_lock(&work_queue_locks[myid]);
current_work = deworkqueue(&work_queues[myid]);
omp_unset_lock(&work_queue_locks[myid]);
runloopchunk(loopid, current_work.lo, current_work.hi);
}
int idx, left_workload;
while (1)
{
get_most_loaded(work_queues, nthreads, &idx, &left_workload);
if (-1 == idx)
{
break;
}
chunk stolen_work;
omp_set_lock(&work_queue_locks[idx]);
if (work_queues[idx].workload != left_workload)
{
omp_unset_lock(&work_queue_locks[idx]);
continue;
}
stolen_work = deworkqueue(&work_queues[idx]);
omp_unset_lock(&work_queue_locks[idx]);
runloopchunk(loopid, stolen_work.lo, stolen_work.hi);
}
#pragma omp barrier
destroy_work_queue(&work_queues[myid]);
omp_destroy_lock(&work_queue_locks[myid]);
}
if (NULL != work_queues)
{
free(work_queues);
}
if (NULL != work_queue_locks)
{
free(work_queue_locks);
}
}
void loop1chunk(int lo, int hi)
{
int i, j;
for (i = lo; i < hi; i++)
{
for (j = N - 1; j > i; j--)
{
a[i][j] += cos(b[i][j]);
}
}
}
void loop2chunk(int lo, int hi)
{
int i, j, k;
double rN2;
rN2 = 1.0 / (double)(N * N);
for (i = lo; i < hi; i++)
{
for (j = 0; j < jmax[i]; j++)
{
for (k = 0; k < j; k++)
{
c[i] += (k + 1) * log(b[i][j]) * rN2;
}
}
}
}
void valid1(void)
{
int i, j;
double suma;
suma = 0.0;
for (i = 0; i < N; i++)
{
for (j = 0; j < N; j++)
{
suma += a[i][j];
}
}
printf("Loop 1 check: Sum of a is %lf\n", suma);
}
void valid2(void)
{
int i;
double sumc;
sumc = 0.0;
for (i = 0; i < N; i++)
{
sumc += c[i];
}
printf("Loop 2 check: Sum of c is %f\n", sumc);
}
void runloopchunk(int loopid, int lo, int hi)
{
switch (loopid)
{
case 1:
loop1chunk(lo, hi);
break;
case 2:
loop2chunk(lo, hi);
break;
}
}