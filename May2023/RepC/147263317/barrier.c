#include <omp.h>
#include <stdio.h>
#include <limits.h>
int main() {
int sum;
#pragma omp parallel num_threads(4)
{
#pragma omp barrier 
printf("yello\n");
#pragma omp barrier 
printf("I'm\n");
#pragma omp barrier 
printf("back\n");
}
return 0;
}
