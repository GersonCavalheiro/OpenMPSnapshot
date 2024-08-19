#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "omp.h"
void long_running_task(int value) {
printf("Thread %d going to sleep for %d seconds\n", omp_get_thread_num(), value);
sleep(value);
printf("Thread %d weaking up after a %d seconds siesta, willing to work ...\n", omp_get_thread_num(), value);
}
void loop_body(int i, int j) {
printf("Thread %d executing loop body (%d, %d)\n", omp_get_thread_num(), i, j);
sleep(1);
}
int main(int argc, char *argv[]) {
#pragma omp parallel num_threads(4)
#pragma omp single
{
printf("I am thread %d and going to create T1 and T2\n", omp_get_thread_num());
#pragma omp task      
long_running_task(5); 
#pragma omp task      
{
#pragma omp task   
long_running_task(10); 
#pragma omp task   
{
#pragma omp taskloop grainsize(1) nogroup  
for (long i = 0; i < 10; i++) 
for (long j = 0; j < i; j++) 
loop_body(i, j);
printf("Thread %d finished the creation of all tasks in taskloop TL\n", omp_get_thread_num());
}
printf("Thread %d finished the execution of task creating T3 and T4\n", omp_get_thread_num());
}
printf("I am still thread %d after creating T1 and T2, ready to enter in the taskwait\n", omp_get_thread_num());
#pragma omp taskwait
printf("I am still thread %d, but now after exiting from the taskwait\n", omp_get_thread_num());
}
return 0;
}
