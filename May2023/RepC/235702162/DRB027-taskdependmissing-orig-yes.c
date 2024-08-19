#include <assert.h> 
#include <stdio.h> 
int main()
{
int i=0;
#pragma omp parallel
#pragma omp single
{
#pragma omp task
i = 1;    
#pragma omp task
i = 2;    
}
printf ("i=%d\n",i);
return 0;
} 
