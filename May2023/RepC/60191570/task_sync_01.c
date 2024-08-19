#include <stdio.h>
void f( )
{
#pragma omp task
{
printf("1");
#pragma omp task
{
printf("2");
#pragma omp task
{
printf("3");
}
#pragma omp barrier
}
}
#pragma omp task
{
printf("4");
#pragma omp task
{
printf("5");
#pragma omp task
{
printf("7");
}
#pragma omp task
{
printf("8");
}
#pragma omp taskwait
}
#pragma omp task
{
printf("6");
}
}
#pragma omp barrier
}
