#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <omp.h>
int main(int argc, char** argv) {
if (argc < 2) {
printf("Provide size of the Matrix! Usage: ./Hadamard n\n");
return EXIT_FAILURE;
}
char* err;
int n = strtol(argv[1], &err, 10);
if (*err != '\0' && n == 0) {
printf("Invalid input! Usage: ./Hadamard n\n");
return EXIT_FAILURE;
}
if (n <= 0) {
printf("Invalid input! Size must be larger than Zero\n");
return EXIT_FAILURE;
}
int32_t (*a)[n] = malloc(sizeof(int[n][n]));
int32_t (*b)[n] = malloc(sizeof(int[n][n]));
int32_t (*c)[n] = malloc(sizeof(int[n][n]));
int threads;
double startTime = omp_get_wtime();
#pragma omp parallel shared(a,b,c, threads) 
{
#pragma omp for
#ifdef COL_MAJOR
for (size_t j = 0;j < n;++j) {
for (size_t i = 0;i < n;++i) {
#else
for (size_t i = 0;i < n;++i) {
for (size_t j = 0;j < n;++j) {
#endif 
c[i][j] = a[i][j] * b[i][j];
}
}
threads = omp_get_num_threads();
}
double endTime = omp_get_wtime();
printf("p = %d, n = %d, t = %2.2f\n", threads, n ,endTime - startTime);
if (a != NULL)
free(a);
if (b != NULL)
free(b);
if (c != NULL)
free(c);
return EXIT_SUCCESS;
}