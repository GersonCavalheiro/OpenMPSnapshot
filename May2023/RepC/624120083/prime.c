#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include "array.h"
#include "prime.h"
array_t* find_primes(int limit) {
bool* sieve = calloc(limit + 1, sizeof(bool));
for (int i = 2; i <= limit; ++i) {
sieve[i] = true;
}
int prime_amount = limit - 1;
for (int i = 2; i * i <= limit; ++i) {
if (sieve[i] == true) {
for (int j = i * i; j <= limit; j += i) {
if (sieve[j]) {
prime_amount--;
}
sieve[j] = false;
}
}
}
int* primes = calloc(prime_amount, sizeof(int));
int idx = 0;
for (int i = 0; i <= limit; ++i) {
if (sieve[i]) {
primes[idx++] = i;
}
}
free(sieve);
array_t* result = create_arr(primes, prime_amount);
free(primes);
return result;
}
array_t* find_primes_parallel(int limit, int n_threads) {
omp_set_num_threads(n_threads);
bool* sieve = calloc(limit + 1, sizeof(bool));
#pragma omp parallel for
for (int i = 2; i <= limit; ++i) {
sieve[i] = true;
}
int prime_amount = limit - 1;
int lim = (int) sqrt(limit);
#pragma omp parallel for schedule(dynamic) shared(prime_amount)
for (int i = 2; i <= lim; ++i) {
if (sieve[i] == true) {
for (int j = i * i; j <= limit; j += i) {
#pragma critical
{
if (sieve[j]) {
prime_amount--;
}
}
sieve[j] = false;
}
}
}
int* primes = calloc(prime_amount, sizeof(int));
int idx = 0;
#pragma omp parallel for
for (int i = 0; i <= limit; ++i) {
if (sieve[i]) {
primes[idx++] = i;
}
}
free(sieve);
array_t* result = create_arr(primes, prime_amount);
free(primes);
return result;
}
int main(int argc, char** argv) {
if (argc == 2) {
printf("Using default number of threads: %d\n", omp_get_num_procs());
}
if (argc < 2) {
fprintf(stderr, "./prime [limit] [n_threads], 2nd arg is optional, defaults to %d\n", omp_get_num_procs());
exit(EXIT_FAILURE);
}
int lim = atoi(argv[1]);
int n_threads = argc == 3 ? atoi(argv[2]) : omp_get_num_procs();
clock_t start = clock();
array_t* primes = find_primes(lim);
clock_t end = clock();
double elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;
printf("Sequential calculation: %lf\n", elapsed);
free_arr(primes);
double start_par = omp_get_wtime();
primes = find_primes_parallel(lim, n_threads);
double end_par = omp_get_wtime();
double elapsed_par = end_par - start_par;
printf("Parallel calculation: %lf\n", elapsed_par);
free_arr(primes);
return EXIT_SUCCESS;
}
