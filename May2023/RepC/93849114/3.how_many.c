#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main ()
{
#pragma omp parallel
printf("Hello world from the first parallel!\n");
omp_set_num_threads(2);
#pragma omp parallel
printf("Hello world from the second parallel!\n");
#pragma omp parallel num_threads(3)
printf("Hello world from the third parallel!\n");
#pragma omp parallel
printf("Hello world from the fourth  parallel!\n");
srand(time(0));
#pragma omp parallel num_threads(rand()%4+1) if(0) 
printf("Hello world from the fifth parallel!\n");
return 0;
}
