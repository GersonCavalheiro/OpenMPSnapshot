#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>
const int L1D_CACHE_SIZE = 32768;
using namespace std;
void segmented_sieve(int64_t limit, int segment_size = L1D_CACHE_SIZE)
{
int sqrt = (int) std::sqrt((double) limit);
vector<char> is_prime(sqrt + 1, 1);
for (int i = 2; i * i <= sqrt; i++)
if (is_prime[i])
for (int j = i * i; j <= sqrt; j += i)
is_prime[j] = 0;
vector<int> primes;
vector<int> next;
int64_t count = (limit < 2) ? 0 : 1;
int64_t s = 2;
int64_t n = 3;
for (int64_t low = 0; low <= limit; low += segment_size)
{
vector<char> segment(segment_size);
fill(segment.begin(), segment.end(), 1);
int64_t high = min(low + segment_size - 1, limit);
for (; s * s <= high; s++)
if (is_prime[s])
{
primes.push_back((int) s);
next.push_back((int)(s * s - low));
}
for (size_t i = 1; i < primes.size(); i++)
{
int j = next[i];
for (int k = primes[i] * 2; j < segment_size; j += k)
segment[j] = 0;
next[i] = j - segment_size;
}
for (; n <= high; n += 2)
count += segment[n - low];
}
cout << endl << count << " primes found." << endl;
}
int main(int argc, char** argv)
{
int64_t limit = 1000000000;
int start, stop;
if (argc >= 2)
limit = atoll(argv[1]);
int size = L1D_CACHE_SIZE;
if (argc >= 3)
size = atoi(argv[2]);
start = clock();
segmented_sieve(limit, size);
stop = clock();
cout << "sieve time: " << (stop-start) << " ms." << endl;
return 0;
}
