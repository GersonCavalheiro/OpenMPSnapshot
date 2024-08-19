#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define B 10000000000L;
void randInit(int array[][2000], int n, int m)
{
srand(time(NULL));
for(int i = 0; i < n; i++)
for(int j = 0; j < m; j++)
array[i][j] = rand() % 100;
}
int main()
{    
constint N = 1000;
constint M = 2000;
int d[N][M];
randInit(d, N, M);
struct timespec start, end;
clock_gettime(CLOCK_REALTIME,&start);
#pragma omp parallel sections
{
#pragma omp section
{
int mean = 0;
printf("Num threads %d\n", omp_get_num_threads());
for(int i = 0; i < N; i++)
for(int j = 0; j < M; j++)
mean += d[i][j]; 
printf("Thread num %d. Arithmetic mean %f\n", omp_get_thread_num(), mean / (float)(N*M));
}
#pragma omp section
{
int min, max;
min = max = d[0][0];
for(int i = 0; i < N; i++)
for(int j = 0; j < M; j++)
{
if(min > d[i][j])
min = d[i][j];
if(max < d[i][j])
max = d[i][j];
}
printf("Thread num %d. min = %d, max = %d\n",omp_get_thread_num(), min, max);
}
#pragma omp section
{
int count = 0;
for(int i = 0; i < N; i++)
for(int j = 0; j < M; j++)
if(d[i][j] % 3== 0)
count++;
printf("Thread num %d.Count=%d\n",omp_get_thread_num(), count);
}
}
clock_gettime(CLOCK_REALTIME,&end);
double accum=(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)/B;
printf("time = %f\n",accum);
return 0;
}
