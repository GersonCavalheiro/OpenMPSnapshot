
#include"stdafx.h"
#include "stdio.h"
#include <conio.h>
#include <time.h>
#include <omp.h>
#define ArraySize 100000000
int *A, *B;
long calc(long istart, long iend)
{
int C = 0;
for (int i = istart; i<iend; i++)
C += A[i] * B[i];
return C;
}

int main()
{
A = new int[ArraySize];
B = new int[ArraySize];
printf("\r\nEntering Main");
for (int i = 0; i<ArraySize; i++)
{
A[i] = 1; B[i] = 1;
}
long t1, t2;
int C1 = 0, C2 = 0, T1 = 0, T2 = 0, T3=0, T4=0, T5=0, T6=0, T7=0, T8=0;
t1 = clock();
C1 = calc(0, ArraySize);
t2 = clock();
printf("\r\nTime for serial code=%d ms ", t2 - t1);
t1 = clock();
#pragma omp parallel num_threads(8)
{
int thr = omp_get_thread_num();
if (thr == 0)
T1 = calc(0, ArraySize / 8);
else if(thr == 1)
T2 = calc(ArraySize / 8, ArraySize / 4);
else if (thr == 2 )
T3 = calc(ArraySize / 4,  3 * ArraySize / 8);
else if (thr == 3 )
T4 = calc( 3 * ArraySize / 8,  ArraySize / 2);
else if (thr == 4)
T5 = calc(4 * ArraySize / 8, 5 * ArraySize / 8);
else if (thr == 5)
T6 = calc(5 * ArraySize / 8, 6 * ArraySize / 8); 
else if (thr == 6)
T7 = calc(6 * ArraySize / 8, 7 * ArraySize / 8);
else if (thr == 7)
T8 = calc(7 * ArraySize / 8, 8 * ArraySize / 8);
}
C2 = T1 + T2 + T3 + T4 + T5 + T6 + T7 + T8;
t2 = clock();
printf("\r\nTime for 2 threads=%d ms ", t2 - t1);
printf("\r\nC1=%d", C1);
printf("\r\nC2=%d", C2);
printf("\r\nExiting Main");
_getch();
return 0;
}
