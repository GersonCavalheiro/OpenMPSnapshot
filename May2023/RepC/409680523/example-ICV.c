#include <omp.h>
#include <stdio.h>
int main()
{
omp_set_nested(1);
omp_set_max_active_levels(8);
omp_set_dynamic(0);
omp_set_num_threads(2); 
#pragma omp parallel
{
omp_set_num_threads(3); 
#pragma omp parallel
{
omp_set_num_threads(8); 
#pragma omp single
{
printf("Inner: max_active_levels=%d, num_threads=%d, max_threads=%d\n", omp_get_max_active_levels(), omp_get_num_threads(), omp_get_max_threads());
}
}
#pragma omp barrier
#pragma omp single
{
printf("Outer: max_active_levels=%d, num_threads=%d, max_threads=%d\n", omp_get_max_active_levels(), omp_get_num_threads(), omp_get_max_threads());
}
}
return 0;
}
