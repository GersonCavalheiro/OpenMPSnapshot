#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 10000
int main(int argc, char **argv)
{
double A[N][N]= {}, X[N]= {}, Y[N]= {}, sum= 0.0, t1= 0.0, t2= 0.0;
int i= 0, j= 0;
if(argc != 2)
{
printf("Usage: %s <seed>\n",argv[0]);
return(1);
}
srand(atoi(argv[1]));
for(i= 0; i < N; i++)
{
X[i]= rand()%1000;
for(j= 0; j < N; j++)
A[i][j]= rand();
}
t1= omp_get_wtime();
#pragma omp parallel num_threads(2)
{
#pragma omp for collapse(2)
for(int i = 0; i < N; i++) {
for (int j = 0; j < N; j++) {
Y[i] += A[i][j] * X[i]; 
}
}
}
t2= omp_get_wtime();
for(i= 0; i < N; i++)
sum+= Y[i];
printf("Sum= %f. Time taken= %f\n",sum,t2-t1);
sum= 0;
for(i= 0; i < N; i++)
Y[i]= 0;
t1= omp_get_wtime();
#pragma omp parallel num_threads(2) reduction(+: Y[:N])
{
#pragma omp for collapse(2) schedule(dynamic, 8000)
for(int i = 0; i < N; i++) {
for(int j = 0; j < N; j++) {
Y[i] += A[j][i] * X[j];
}
}
}
t2= omp_get_wtime();
for(i= 0; i < N; i++)
sum+= Y[i];
printf("Sum= %f. Time taken= %f\n",sum,t2-t1);
}
