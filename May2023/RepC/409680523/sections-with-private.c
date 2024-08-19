#include <stdio.h>
#include <omp.h>
#define NT 4
int main(void)
{
int section_count=0;
omp_set_dynamic(0);
omp_set_num_threads(NT);
#pragma omp parallel firstprivate(section_count)
#pragma omp sections
{
#pragma omp section
{
section_count++; 
printf("section count %d\n", section_count);
}
#pragma omp section
{
section_count++; 
printf("section count %d\n", section_count);
}
}
return 0;
}