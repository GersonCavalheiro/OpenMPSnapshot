#include <omp.h>
#include <stdio.h>
#define NT 4
int main( ) {
int section_count = 0;
omp_set_dynamic(0);
omp_set_num_threads(NT);
#pragma omp parallel
#pragma omp sections firstprivate( section_count )
{
#pragma omp section
{
section_count++;
printf( "section_count %d\n", section_count );
}
#pragma omp section
{
section_count++;
printf( "section_count %d\n", section_count );
}
}
return 1;
}
