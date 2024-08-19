#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main()
{
int i;
#pragma omp parallel shared(i)
{
int thread = omp_get_thread_num();

if (thread == 0)
{
sleep(0);
}
if (thread == 1)
{
sleep(1);
}
if (thread == 2)
{
sleep(4);
}
if (thread == 3)
{
sleep(5);
}
printf("Thread %i before critical\n", thread);
#pragma omp critical
{
if (thread == 0)
{
sleep(2);
}
if (thread == 1)
{
usleep(10);
}
if (thread == 2)
{
sleep(2);
}
if (thread == 3)
{
usleep(10);
}
}
printf("Thread %i after critical\n", thread);
}
}
