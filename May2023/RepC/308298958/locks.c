#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#define N UINT_MAX
int main(){
unsigned char *A;
unsigned int i, count;
time_t start_t, end_t;
double total;
A = (unsigned char*) malloc(N * sizeof(unsigned char));
printf("Filling Array with Random Vaues...\n");
srand(time(NULL));
for(i = 0; i < N; i++){
A[i] = rand() % 256;
}
printf("Starting Sequential Calculation...\n");
count = 0;
start_t = time(NULL);
for (i = 0; i < N; i++){
if(A[i] == 0){
count++;
}
}
end_t = time(NULL);
total = difftime(end_t, start_t);
printf("Sequential Execution Time: %.2f sec, Zeroes in Array: %u or %.2f%%\n",
total, count, (double)count/UINT_MAX * 100);
omp_lock_t lock;
omp_init_lock(&lock);
printf("Starting Parallel Calculation...\n");
count = 0;
start_t = time(NULL);
#pragma omp parallel for private(i)
for (i = 0; i < N; i++){
if(A[i] == 0){
omp_set_lock(&lock);
count++;
omp_unset_lock(&lock);
}
}
end_t = time(NULL);
total = difftime(end_t, start_t);
printf("Parallel Execution Time: %.2f sec, Zeroes in Array: %u or %.2f%%\n",
total, count, (double)count/UINT_MAX * 100);
omp_destroy_lock(&lock);
free(A);
return 0;
}
