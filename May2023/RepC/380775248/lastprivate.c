#include <stdio.h>
#include <stdlib.h>
#include "omp.h"
#include "time.h"



int main(void) {

int i, j, A[4][6] = {0}, sum = 0;

int cnt = 1;
for (i = 0; i < 4; i++)
for (j = 0; j < 6; j++)
A[i][j] = cnt++;

int rowSum[4] = {0};
#pragma omp parallel private(i, j) shared(A, sum) 
{ 
#pragma omp for schedule(static)
for (i = 0; i < 4; i++)
for (j = 0; j < 6; j++)
rowSum[i] += A[i][j];

#pragma omp for schedule(static)
for (i = 0; i < 4; i++)
sum += rowSum[i];

} 


printf("sum = %d\n", sum);

}