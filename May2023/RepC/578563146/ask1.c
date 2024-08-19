#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv){

int numCPU, nthreads, id;
clock_t cpuStart, cpuEnd;

numCPU = sysconf(_SC_NPROCESSORS_ONLN);
printf("Number of processors: %d\n", numCPU);

printf("Maximum number of threads: %d\n", omp_get_max_threads());

cpuStart = clock();

#pragma omp parallel private(nthreads, id)
{
id = omp_get_thread_num();
printf("Hi there! I am thread  %d\n", id);

#pragma omp master
{
printf("Hello. I am the master thread.\n");

nthreads = omp_get_num_threads();
printf("Number of threads participating in execution: %d\n", nthreads);

}
}

cpuEnd = clock();

double cpuElapsedTime = ( (double)(cpuEnd - cpuStart) ) / CLOCKS_PER_SEC ;
printf("CPU Elapsed Time: %10.8f sec\n", cpuElapsedTime);

return 0;
}
