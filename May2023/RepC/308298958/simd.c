#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#define N UINT_MAX 
int main(){
unsigned short *A, *B, *C;
unsigned int i;
double start, end, total;
A = (unsigned short *) calloc(N, sizeof(unsigned short));
B = (unsigned short *) calloc(N, sizeof(unsigned short));
C = (unsigned short *) calloc(N, sizeof(unsigned short));
printf("Filling Arrays with Random Byte Values...\n");
srand(time(NULL));
for(i = 0; i < N; i++){
A[i] = rand() % 256;
B[i] = rand() % 256;
}
printf("Starting Serial Processing..\n");
start = omp_get_wtime();
for (i = 0; i < N; i++){
C[i] = A[i] * B[i];
}
end = omp_get_wtime();
total = end - start;
printf("Serial Execution Time: %.2f sec\n\n", total);
printf("Filling Arrays with Random Byte Values...\n");
srand(time(NULL));
for(i = 0; i < N; i++){
A[i] = rand() % 256;
B[i] = rand() % 256;
}
printf("Starting SIMD Processing..\n");
start = omp_get_wtime();
#pragma omp simd
for (i = 0; i < N; i++){
C[i] = A[i] * B[i];
}
end = omp_get_wtime();
total = end - start;
printf("SIMD Execution Time: %.2f sec\n\n", total);
printf("Filling Arrays with Random Byte Values...\n");
srand(time(NULL));
for(i = 0; i < N; i++){
A[i] = rand() % 256;
B[i] = rand() % 256;
}
printf("Starting Parallel For Processing..\n");
start = omp_get_wtime();
#pragma omp parallel for private(i)
for (i = 0; i < N; i++){
C[i] = A[i] * B[i];
}
end = omp_get_wtime();
total = end - start;
printf("Parallel For Execution Time: %.2f sec\n\n", total);
printf("Filling Arrays with Random Byte Values...\n");
srand(time(NULL));
for(i = 0; i < N; i++){
A[i] = rand() % 256;
B[i] = rand() % 256;
}
printf("Starting Parallel For Simd Processing..\n");
start = omp_get_wtime();
#pragma omp parallel for simd private(i)
for (i = 0; i < N; i++){
C[i] = A[i] * B[i];
}
end = omp_get_wtime();
total = end - start;
printf("Parallel For Simd Execution Time: %.2f sec\n", total);
free(A);
free(B);
free(C);
return 0;
}
