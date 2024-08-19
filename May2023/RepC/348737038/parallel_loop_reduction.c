#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
int i, n, tid;
printf("******* 18BCI0174 ARYAN *******\n");
int a[100], b[100], sum;
n = 12;
for (i = 0; i < n; i++)
a[i] = b[i] = i * 1;
sum = 0;
#pragma omp parallel for reduction(+ : sum)
for (i = 0; i < n; i++)
{
tid = omp_get_thread_num();
sum = sum + (a[i] * b[i]);
printf("[~] thread = %d holds value=%d\n", tid, i);
}
printf("\n\t[+] Sum of all value's square = %d\n", sum);
}
