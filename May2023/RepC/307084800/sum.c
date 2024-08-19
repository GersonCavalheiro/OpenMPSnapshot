#include <omp.h>
#include <stdio.h>
int main(int argc, char **argv) {
int N = atoi(argv[1]);
int sum = 0;
#pragma omp parallel
{	
int local_sum = 0;
#pragma omp for
for (int i = 1; i <= N; i++) {
local_sum +=i;
}
#pragma omp atomic
sum += local_sum;
}
printf("%d\n", sum);
}