#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include "../utils/utils_sort.h"
#include "../utils/utils_1d_array.h"
#ifndef SIEVE_MAX_VALUE
#define SIEVE_MAX_VALUE 100
#endif
#ifndef OMP_THREADS
#define OMP_THREADS 1
#endif
FILE *fp;
int get_sqrt(int value)
{
int res = 0;
while ((res + 1) * (res + 1) <= value)
{
res += 1;
}
return res;
}
void filter_primes_simple(int limit, int *primes)
{
int count = 0;
int isPrime = 0;
int sqrLimit = get_sqrt(limit);
int arr[limit];
for (int i = 0; i < limit; i++)
{
arr[i] = 1;
}
for (int i = 2; i < sqrLimit; i++)
{
if (arr[i] == 1)
{
for (int j = i * i; j < limit; j = j + i)
{
arr[j] = 0;
}
}
}
for (int i = 0; i < limit; i++)
{
if (arr[i] == 1)
{
primes[count] = i;
count++;
}
}
}
void filter_primes_omp(int limit, int *primes)
{
int count = 0;
int isPrime = 0;
int sqrLimit = get_sqrt(limit);
int arr[limit];
#pragma omp parallel for schedule(static, OMP_THREADS) num_threads(OMP_THREADS)
for (int i = 0; i < limit; i++)
{
arr[i] = 1;
}
#pragma omp parallel for schedule(static, OMP_THREADS) num_threads(OMP_THREADS)
for (int i = 2; i < sqrLimit; i++)
{
if (arr[i] == 1)
{
for (int j = i * i; j < limit; j = j + i)
{
arr[j] = 0;
}
}
}
for (int i = 0; i < limit; i++)
{
if (arr[i] == 1)
{
primes[count] = i;
count++;
}
}
}
int main(void)
{
int max_val = SIEVE_MAX_VALUE;
struct timeval begin, end;
int arr_orig[SIEVE_MAX_VALUE];
int arr_omp[SIEVE_MAX_VALUE];
for (int i = 0; i < SIEVE_MAX_VALUE; i++)
{
arr_orig[i] = 0;
arr_omp[i] = 0;
}
filter_primes_simple(SIEVE_MAX_VALUE, arr_orig);
gettimeofday(&begin, 0);
filter_primes_omp(SIEVE_MAX_VALUE, arr_omp);
gettimeofday(&end, 0);
long seconds = end.tv_sec - begin.tv_sec;
long microseconds = end.tv_usec - begin.tv_usec;
double time_spent = seconds + microseconds * 1e-6;
int equal = is_equal(arr_omp, arr_orig, SIEVE_MAX_VALUE);
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Sieve of eratostenes, %d, %d, %f, %d\n", SIEVE_MAX_VALUE, OMP_THREADS, time_spent, equal);
fclose(fp);
return 0;
}