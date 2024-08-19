#include <stdio.h> 
#include <omp.h>
int main( ) 
{
int i;
#pragma omp parallel
{
#pragma omp critical (A)
{
printf("In critical");
#pragma omp critical (B)
{
}
}
}
}
