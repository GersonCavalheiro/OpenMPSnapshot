#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel num_threads(4)
{
#pragma omp single 
printf("yello\n");
printf("I'm\n");
printf("back\n");
}
return 0;
}
