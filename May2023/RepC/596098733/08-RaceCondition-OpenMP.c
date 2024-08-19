#include <stdio.h>
#include <omp.h>
#define N 1000000
int main(int argc, char *argv[]) {
int i;
int sum = 0;
int num_threads = omp_get_num_procs();
printf("Number of cores: %d\n", num_threads);
#pragma omp parallel for num_threads(num_threads) shared(sum)
for (i = 0; i < N; i++) {
#pragma omp critical
{
sum += i;
}
}
printf("Sum: %d\n", sum);
return 0;
}
