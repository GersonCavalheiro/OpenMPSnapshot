#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main (int argc, char *argv[])
{
int A[10][10];
int B[10][10];
int C[10][10];
double t1, t2;
t1 = omp_get_wtime ( );
printf("Done by Maitreyee\n\n\n");
for(int i = 0; i< 10 ; i++)
{
for(int j =0; j<10; j++)
{
A[i][j] = rand();
B[i][j] = rand();
}
}
printf("Matrix A: \n");
for(int i = 0; i<10;i++)
{
for(int j = 0; j<10; j++)
{
printf("%d   ", A[i][j]);
}
printf("\n");
}
printf("\n\n");
printf("Matrix B: \n");
for(int i = 0; i<10;i++)
{
for(int j = 0; j<10; j++)
{
printf("%d   ", B[i][j]);
}
printf("\n");
}
printf("\n\n");
#pragma omp parallel num_threads(4)
{
#pragma omp for collapse(3)
for(int i=0;i<10;i++){
for(int j=0;j<10;j++){
for(int k=0;k<10;k++){
#pragma omp atomic
C[i][j] = C[i][j]+ (A[i][k] * B[k][j]);
}
}
}
}
printf("Matrix C = A*B: \n");
for(int i = 0; i<10;i++)
{
for(int j = 0; j<10; j++)
{
printf("%d   ", C[i][j]);
}
printf("\n");
}
printf("\n\n");
t2 = omp_get_wtime ( ) - t1;
printf("Execution time: %8f\n", t2);
return 0;
}
