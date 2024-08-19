#include <stdio.h>
#include <omp.h>
void task1()
{
printf("Executing task 1 on thread %d\n", omp_get_thread_num());
}
void task2()
{
printf("Executing task 2 on thread %d\n", omp_get_thread_num());
}
void task3()
{
printf("Executing task 3 on thread %d\n", omp_get_thread_num());
}
int main()
{
#pragma omp parallel
{
#pragma omp single
{
#pragma omp task
{
task1();
}
#pragma omp task
{
task2();
}
#pragma omp task
{
task3();
}
}
}
return 0;
}
