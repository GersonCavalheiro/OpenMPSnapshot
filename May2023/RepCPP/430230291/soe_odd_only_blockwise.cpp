#include "misc.hpp"


int sieveProcessOddBlock(long long l, long long r) {
const int size = (r - l + 1) >> 1;

char* isPrime = new char[size];
for (int i = 0; i < size; i++) isPrime[i] = 1;

for (uint i = 3; i * 1LL * i <= r; i += 2) {
if (i >= 3 * 3 && i % 3 == 0) continue;
if (i >= 5 * 5 && i % 5 == 0) continue;
if (i >= 7 * 7 && i % 7 == 0) continue;
if (i >= 11 * 11 && i % 11 == 0) continue;
if (i >= 13 * 13 && i % 13 == 0) continue;

long long start = ((l + i - 1) / i) * i;
if (start < i * 1LL * i) start = i * 1LL * i;
if ((start & 1) == 0) start += i; 
for ( ; start <= r; start += i << 1) isPrime[(start - l) >> 1] = 0;
}

int result = 0;
for (int i = 0; i < size; i++) result += isPrime[i];
if (l <= 2) result++;

delete[] isPrime;
return result;
}

long long sieveOddBlockwise(long long N, bool isOpenMP) {
omp_set_num_threads(isOpenMP ? omp_get_num_procs() : 1);
long long result = 0;
#pragma omp parallel for reduction(+:result)
for (long long l = 2; l <= N; l += BLOCK_SIZE) {
long long r = l + BLOCK_SIZE;
if (r > N) r = N;
result += sieveProcessOddBlock(l, r);
}
return result;
}