#include "misc.hpp"

long long sieve(long long N, bool isOpenMP) {
omp_set_num_threads(isOpenMP ? omp_get_num_procs() : 1);

const uint sqrtN = (uint)sqrtl((long double)N);

char* isPrime = new char[N + 1];

#pragma omp parallel for
for (uint i = 0; i <= N; i++) isPrime[i] = 1;

#pragma omp parallel for schedule(dynamic)
for (uint i = 2; i <= sqrtN; i++) {
if (isPrime[i]) {
for (long long j = i * 1LL * i; j <= N; j += i) isPrime[j] = 0;
}
}

long long result = 0;
#pragma omp parallel for reduction(+:result)
for (uint i = 2; i <= N; i++) result += isPrime[i];

delete[] isPrime;
return result;
}

std::vector < uint > findOddPrimes(uint N) {
char *isPrime = new char[N + 1];
for (uint i = 0; i <= N; i++) isPrime[i] = 1;
for (uint i = 2; i <= N; i++) {
if (isPrime[i]) {
if (i * 1LL * i <= N) {
for (uint j = i * i; j <= N; j += i) isPrime[j] = 0;
}
}
}
std::vector < uint > result;
for (uint i = 3; i <= N; i++) if (isPrime[i]) result.push_back(i);

delete[] isPrime;
return result;
}