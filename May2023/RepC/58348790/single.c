#include <stdio.h>
#include <omp.h>
int main() {
#pragma omp parallel num_threads(2)
{
#pragma omp single
{
printf("read input\n");
}
printf("compute results\n");
#pragma omp barrier
#pragma omp single
{
printf("write output\n");
}
}
}
