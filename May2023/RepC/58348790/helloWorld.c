#include<stdio.h>
#include<omp.h>
int main()
{
int ID;
#pragma omp parallel  default (none) private(ID)
{
ID = omp_get_thread_num();
printf ("hello(%d)", ID);
printf ("world(%d)\n", ID);
#pragma omp barrier
if(ID == 0 ) {
int nThreads = omp_get_num_threads ();  
printf ("\n\nWe are %d in number\n\n", nThreads);
}
}
return 0;
}
