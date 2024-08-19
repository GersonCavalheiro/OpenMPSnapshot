#include <stdio.h>
#include <omp.h>
#define NUM_THREADS 4
int shared_counter = 0;
void increment_counter() {
#pragma omp atomic
shared_counter++;
}
int main() {
#pragma omp parallel num_threads(NUM_THREADS)
{
for (int i = 0; i < 1000; i++) {
increment_counter();
}
}
printf("Final value of shared counter: %d\n", shared_counter);
return 0;
}
