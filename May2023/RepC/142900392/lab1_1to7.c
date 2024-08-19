#include <omp.h>
#include<stdio.h>
int main()
{
printf("==========EXERCISES 1 TO 7==========\n");
printf("	 Basic OpenMP Constructs	");
printf("\n\t\t\t\tS.DHANYA ABHIRAMI \n\t\t\t\t16BCE0965 \n\n");
if(omp_in_parallel()==0)
printf("In Serial\n");
else
printf("In Parallel\n");
int max_threads=omp_get_max_threads();
printf("Maximum Number of Threads: %d\n",max_threads);
int thread_limit=omp_get_thread_limit();
printf("Maximum Value for setting: %d\n",thread_limit);
int init_threads=omp_get_num_threads();
printf("Initial Number of Threads: %d\n\n",init_threads);
int num_threads;
printf("Enter number of threads to be set: ");
scanf("%d",&num_threads);
omp_set_num_threads(num_threads);
double time_before,time_after;
time_before=omp_get_wtime();
#pragma omp parallel
{
if(omp_in_parallel()==0)
printf("In Serial\n");
else
printf("In Parallel\n");
int num_process=omp_get_num_procs();
printf("Number of Processess: %d\n",num_process);		
int ID = 0;
printf("hello(%d)", ID);
printf("world(%d) \n", ID);
int num1=omp_get_num_threads();
printf("Number of Threads: %d\n",num1);
int num=omp_get_thread_num();
printf("Thread ID: %d\n\n",num);
}
time_after=omp_get_wtime();
printf("Time: %lf\n",time_after-time_before);
return 0;
}