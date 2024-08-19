#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void Hello(char *st)
{
int my_rank, nthreads, nmaxthreads;
nmaxthreads = omp_get_max_threads () ;
nthreads = omp_get_num_threads();
my_rank = omp_get_thread_num();
printf("%s thread %d of team %d (max_num_threads is %d)\n", st, my_rank, nthreads,nmaxthreads);
} 
int main (int argc, char*argv[])  
{
int nthreads;
if (argc != 2)
{
fprintf (stderr, "the parameter (thread number) is missing!\n") ;
fprintf (stderr, "exo1 thread_number\n") ;
exit (-1) ;
}
nthreads = atoi (argv[1]);
printf("I am the master thread %d and I start\n", omp_get_thread_num ());
printf ("Starting Region 1 \n") ;
#pragma omp parallel num_threads (nthreads)
Hello("Region 1 ") ; 
printf ("End of Region 1\n") ;
printf ("Starting Region 2 \n") ;
#pragma omp parallel num_threads (nthreads/2) 
Hello ("Region 2 ") ;
printf ("End of Region 2\n") ;
printf ("Starting Region 3 \n") ;
#pragma omp parallel num_threads (nthreads/4) 
Hello ("Region 3 ") ;
printf ("End of Region 3\n") ;
printf("I am the master thread %d and I complete\n", omp_get_thread_num ());
return 0;
} 
