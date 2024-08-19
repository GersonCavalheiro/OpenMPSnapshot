#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int a[11], b[11], sum;
int dotprod()
{
int i, tid;
tid = omp_get_thread_num();
#pragma omp for reduction(+ : sum)
for (i = 0; i < 11; i++)
{
sum = sum + (a[i] * b[i]);
printf("[~] thread = %d holds index = %d\n", tid, i);
}
}
int main(int argc, char *argv[])
{
printf("******** 18BCI0174 ARYAN *******\n");
int i;
for (i = 0; i < 11; i++)
a[i] = b[i] = 1 * i;
sum = 0;
#pragma omp parallel
dotprod();
printf("\n\t[+] Sum of squares of values = %d\n", sum);
}
