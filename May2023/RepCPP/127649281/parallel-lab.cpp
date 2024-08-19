#include "stdafx.h"


#pragma optimize( "2", on )
#include <stdio.h>
#include <windows.h>
#include <mmsystem.h>
#include <omp.h>
#include <math.h>

#pragma comment(lib, "winmm.lib")
const long int VERYBIG = 100000;
const int NUM_THREADS = 8;

int main(void)
{
int i;
long int j, k, sum;
double total;
DWORD starttime, elapsedtime;
printf("Parallel Timings for %d iterations\n\n", VERYBIG);

omp_set_num_threads(NUM_THREADS);

for (i = 0; i<6; i++)
{
starttime = timeGetTime();
sum = 0;
total = 0.0;

#pragma omp parallel for reduction(+:sum, total) schedule(static, 1000)
for (j = 0; j<VERYBIG; j++)
{
sum += 1;
double sumx = 0.0;
#pragma omp parallel for reduction(+:sumx)
for (k = 0; k<j; k++) 
sumx += (double)k;
double sumy = 0.0;
#pragma omp parallel for reduction(+:sumy)
for (k = j; k>0; k--)
sumy += (double)k;
if (sumx > 0.0)total += 1.0 / sqrt(sumx);
if (sumy > 0.0)total += 1.0 / sqrt(sumy);
}
elapsedtime = timeGetTime() - starttime;
printf("Time Elapsed % 10d mSecs Total = %lf Check Sum = %ld\n",
(int)elapsedtime, total, sum);
}


return 0;
}
