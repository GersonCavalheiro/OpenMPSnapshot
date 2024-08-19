#include<stdio.h>
int main( int ac, char **av)
{
#pragma omp parallel 
{
printf("Hello World!!!\n");
}
return 0;
}
