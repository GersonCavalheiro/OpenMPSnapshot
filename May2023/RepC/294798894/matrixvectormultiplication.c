#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>
#define SIZE 10
int A[SIZE][SIZE];
int b[SIZE];
int c[SIZE];
int total;
int tid;
int main()
{
printf("Maitreyee Paliwal\n\n");
int i,j,k;
omp_set_num_threads(omp_get_num_procs());
for (i= 0; i< SIZE ; i++)
for (j= 0; j< SIZE ; j++)
{
A[i][j] = 4;
b[i] = 4;
}
#pragma omp parallel num_threads(4)
{
int  y;
int i; 
#pragma omp for schedule(static,16)
for(y = 0; y < SIZE ; y++){
double output = 0;
for(i = 0; i < SIZE; i++){
output += b[i]*A[i][y];
}
c[y] = output;
}
for(y=0; y<SIZE; y++)
{
printf("%d ", c[y]);
}
}
}
