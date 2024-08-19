#include <stdio.h>;
#include <omp.h>;
void main()
{
omp_set_dynamic( 0 );
omp_set_num_threads( omp_num_procs() );
#pragma omp parallel	
{
printf("hello(%d)",5);
printf("word(%d) \n",5);
}
}
