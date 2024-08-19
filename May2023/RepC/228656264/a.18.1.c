#include <omp.h>
#include <stdio.h>
extern void abort (void);
#define NUMBER_OF_THREADS 4
int synch[NUMBER_OF_THREADS];
int work[NUMBER_OF_THREADS];
int result[NUMBER_OF_THREADS];
int
fn1 (int i)
{
return i * 2;
}
int
fn2 (int a, int b)
{
return a + b;
}
int
main ()
{
int i, iam, neighbor;
omp_set_num_threads (NUMBER_OF_THREADS);
#pragma omp parallel private(iam,neighbor) shared(work,synch)
{
iam = omp_get_thread_num ();
synch[iam] = 0;
#pragma omp barrier
work[iam] = fn1 (iam);
#pragma omp flush(work,synch)
synch[iam] = 1;
#pragma omp flush(synch)
neighbor = (iam > 0 ? iam : omp_get_num_threads ()) - 1;
while (synch[neighbor] == 0)
{
#pragma omp flush(synch)
}
#pragma omp flush(work,synch)
result[iam] = fn2 (work[neighbor], work[iam]);
}
for (i = 0; i < NUMBER_OF_THREADS; i++)
{
neighbor = (i > 0 ? i : NUMBER_OF_THREADS) - 1;
if (result[i] != i * 2 + neighbor * 2)
abort ();
}
return 0;
}
