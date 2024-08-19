#include <stdio.h>
#include <omp.h>
extern void abort (void);
int
main ()
{
int bad, x;
x = 2;
bad = 0;
#pragma omp parallel num_threads(2) shared(x, bad)
{
if (omp_get_thread_num () == 0)
{
volatile int i;
for (i = 0; i < 100000000; i++)
x = 5;
}
else
{
if (x != 2 && x != 5)
bad = 1;
}
#pragma omp barrier
if (omp_get_thread_num () == 0)
{
if (x != 5)
bad = 1;
}
else
{
if (x != 5)
bad = 1;
}
}
if (bad)
abort ();
return 0;
}
