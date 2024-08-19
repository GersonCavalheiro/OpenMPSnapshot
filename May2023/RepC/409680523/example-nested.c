#include <stdio.h>
#include <omp.h>
int main(void)
{
omp_set_nested(1); 
omp_set_dynamic(0); 
#pragma omp parallel
{
#pragma omp parallel
{
#pragma omp single
printf("Inner: num_threads=%d\n",omp_get_num_threads());
}
#pragma omp barrier
omp_set_nested(0);
#pragma omp parallel
{
#pragma omp single
printf("Inner: num_threads=%d\n",omp_get_num_threads());
}
#pragma omp barrier
#pragma omp single
printf("Outer: num_threads=%d\n",omp_get_num_threads());
}
return 0;
}