#include "bits/stdc++.h"
#include "omp.h"
using namespace std;
#define SIZE 65536
#define a 10
void serial(int x[],int y[])
{
int i;
for(i=0;i<SIZE;i++)
{
x[i]=a*x[i]+y[i];
}
}
void parallel(int x[],int y[],int number_of_threads)
{
int i;
omp_set_num_threads(number_of_threads);
int n_per_thread = SIZE/number_of_threads;
#pragma omp parallel for shared(x,y) private(i)
for(i=0;i<SIZE;i++)
{
x[i]=a*x[i]+y[i];
}	
}
int main()
{
int x[SIZE],y[SIZE],i,j;
for(i=0;i<SIZE;i++)
{
x[i]=rand()%1000;
}
for(i=0;i<SIZE;i++)
{
y[i]=rand()%1000;
}
printf("Threads\tSerial Time\tParallel Time\tSpeedUp\n");
for(i=2;i<10;i++)
{
double serial_start_time, serial_end_time;
double parallel_start_time, parallel_end_time;
serial_start_time = omp_get_wtime();
serial(x,y);
serial_end_time = omp_get_wtime();
parallel_start_time = omp_get_wtime();
parallel(x,y,i);
parallel_end_time = omp_get_wtime();
printf("%d\t%lf\t%lf\t%lf\n", i, (serial_end_time - serial_start_time)*1000,(parallel_end_time - parallel_start_time)*1000, (parallel_end_time - parallel_start_time)/(serial_end_time - serial_start_time));
}
}
