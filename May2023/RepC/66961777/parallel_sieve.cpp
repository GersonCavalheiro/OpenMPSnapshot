#include <iostream>
#include <math.h>
#include <string.h>
#include <omp.h>
using namespace std;
double t;
long mark_multiples(bool prime[], long low, long multiple, long high) {
long i;
for (i=low; i <= high; i += multiple) {
prime[i] = false;
}
return i;
}
long CacheSieve(long N) {
long count = 0;
bool* prime = new bool[N+1];
for (long p=2; p<=N; p++) {
prime[p] = true;
}
long M = (long) sqrt((double) N);
long* factor  = new long[M];
long* striker = new long[M];
long n_factor = 0;
double t1 = omp_get_wtime();
for (long p=2; p <= M; p++) {
if (prime[p] == true) {
count++;
striker[n_factor] = mark_multiples(prime, p*2, p, M);
factor[n_factor++] = p;
}
}
for (long window = M+1; window <= N; window += M) {
long limit = min(window + M - 1, N); 
for (long p=0; p < n_factor; p++) {
striker[p] = mark_multiples(prime, striker[p], factor[p], limit);
}
for (long p=window; p<=limit; p++) {
if (prime[p] == true) {
count++;
}
}
}
double t2 = omp_get_wtime();
t = (t2 - t1) * 1000;
return count;
}
long ParallelSieve(long N) {
long count = 0;
long M = (long) sqrt((double) N);
long* factor  = new long[M];
long n_factor = 0;
double t1 = omp_get_wtime();
#pragma omp parallel
{
bool* prime = new bool[M+1];
long* striker = new long[M];
#pragma omp single
{
memset(prime, true, M);
for (long p=2; p <= M; p++) {
if (prime[p] == true) {
count++;
mark_multiples(prime, 2*p, p, M);
factor[n_factor++] = p;
}
}
}
long base = -1;
#pragma omp for reduction(+:count)
for (long window = M+1; window <= N; window += M) {
memset(prime, true, M);
if (base != window) {
base = window;
for(long k=0; k < n_factor; k++ ) {
striker[k] = (base + factor[k] - 1) / factor[k] * factor[k] - base;
}
}
long limit = min(window + M -1, N) - base;
for(long k=0; k<n_factor; k++ ) {
striker[k] = mark_multiples(prime, striker[k], factor[k], limit) - M;
}
for (long p=0; p <= limit; p++) {
if (prime[p] == true) {
count++;
}
}
base += M;
}
}
double t2 = omp_get_wtime();
t = (t2 - t1) * 1000;
return count;
}
int main(int argc, char* argv[]) {
long N = atol(argv[1]);
int num_threads = atoi(argv[2]);
omp_set_num_threads(num_threads);
long count_cache = CacheSieve(N);
cout << "Cache Friendly Sieve -> Time: " << t << "ms, Count = " << count_cache << endl;
long count_parallel = ParallelSieve(N);
cout << "Parallel Sieve -> Time: " << t << "ms, Count = " << count_parallel << endl;
}
