#include <omp.h>
#include <stdio.h>
int main() {
int sum;
#pragma omp parallel reduction(+: sum) num_threads(8)
{
sum += 10;
}
printf("%d\n", sum);
return 0;
}
