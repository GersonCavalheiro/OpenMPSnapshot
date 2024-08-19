#include<stdio.h>
#include<omp.h>
int pooh(double A[], int ID)
{
printf ("ID = %d\n", ID);
}
void main()
{
double A[1000];
#pragma omp parallel num_threads(6)
{
int ID = omp_get_thread_num();
pooh(A,ID);	
}
printf("All Done\n");
}
