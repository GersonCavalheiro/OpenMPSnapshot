#include <omp.h>
#include <stdio.h>
int main(int argc, char **argv) {
int N = atoi(argv[1]);
int sum = 0;
#pragma omp parallel for schedule(static)
for (intptr_t i = 0; i <= N; i++) {
sum = sum + i;
}
printf("%d\n", sum);
}