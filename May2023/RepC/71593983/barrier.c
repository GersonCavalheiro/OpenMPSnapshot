#include<stdio.h>
#include<omp.h>
int main()
{
int s = 0, i, n = 5;
int extra = 2;
#pragma omp parallel
{
#pragma omp for reduction(+:s)
for(i = 0; i < n; i++)
{	
s = s + 1;
}
#pragma omp critical 
s = s + extra; 
#pragma omp barrier
printf("s = %i \n", s);
}
return 0;
}
