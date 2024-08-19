#include <omp.h>
#include <stdio.h>
#define N 1000
int main() {
int sum = 0;
#pragma omp parallel for shared(sum) num_threads(4)
for(int i = 0; i < N; i++) {
#pragma omp atomic 
sum++;
}
printf("%d\n", sum);
return 0;
}
