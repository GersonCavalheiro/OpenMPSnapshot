#include "collatz.h"
void collatz(int startNumber, int endNumber, int* iter, int nThreads)
{
int i, n, counter;
int isodd; 
#pragma omp parallel for private(counter, n, isodd) 
for (i = startNumber; i <= endNumber; i++)
{
counter = 0;
n = i;
omp_set_num_threads(nThreads);
while (n > 1)
{
isodd = n%2;
if (isodd)
n = 3*n+1;
else
n/=2;
counter++;
}
iter[i - startNumber] = counter;
}
}