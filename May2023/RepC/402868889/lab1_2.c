#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
const int count = 10000000;            
const int threads = atoi(argv[1]);     
const int random_seed = atoi(argv[2]); 
int *array = 0;                        
int max = -1;                          
int iterations;                        
srand(random_seed);
printf("===========================================================\nOpenMP: %d\n", _OPENMP);
array = (int *)malloc(count * sizeof(int));
for (int i = 0; i < count; i++) {
array[i] = rand();
}
#pragma omp parallel num_threads(threads) private(iterations) shared(array, count) reduction(max: max) default(none)
{
iterations = 0;
#pragma omp for
for (int i = 0; i < count; i++) {
if (array[i] > max) {
max = array[i];
}
iterations++;
}
printf("local maximum: %d || iterations performed: %d\n", max, iterations);
}
printf("total maximum: %d\n===========================================================\n", max);
return 0;
}