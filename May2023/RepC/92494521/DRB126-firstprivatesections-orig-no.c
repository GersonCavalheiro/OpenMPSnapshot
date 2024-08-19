#include <omp.h>
#include <stdio.h>
int main(){
int section_count = 0;
omp_set_dynamic(0);
omp_set_num_threads(1);
#pragma omp parallel
#pragma omp sections firstprivate( section_count )
{
#pragma omp section
{
section_count++;
printf("%d\n",section_count);
}
#pragma omp section
{
section_count++;
printf("%d\n",section_count);
}
}
return 0;
}
