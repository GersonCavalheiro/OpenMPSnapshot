#include "misc.hpp"


long long sieveOdd(long long N, bool isOpenMP) {
omp_set_num_threads(isOpenMP ? omp_get_num_procs() : 1);

const uint sqrtN = (uint)sqrtl((long double)N);

const long long size = (N - 1) / 2;

char* isPrime = new char[size + 1];

#pragma omp parallel for
for (uint i = 0; i <= size; i++) isPrime[i] = 1;

#pragma omp parallel for schedule(dynamic)
for (uint i = 3; i <= sqrtN; i += 2) {
if (isPrime[i >> 1]) {
for (long long j = i * 1LL * i; j <= N; j += i << 1) isPrime[j >> 1] = 0;
}
}

long long result = N >= 2 ? 1 : 0;
#pragma omp parallel for reduction(+:result)
for (uint i = 1; i <= size; i++) result += isPrime[i];

delete[] isPrime;
return result;
}