
#include <gourdon.hpp>
#include <primecount-internal.hpp>
#include <primesieve.hpp>
#include <int128_t.hpp>
#include <LoadBalancerP2.hpp>
#include <macros.hpp>
#include <min.hpp>
#include <imath.hpp>
#include <print.hpp>

#include <stdint.h>
#include <algorithm>

using namespace primecount;

namespace {

template <typename T>
T B_thread(T x,
int64_t y,
int64_t low,
int64_t high)
{
ASSERT(low > 0);
ASSERT(low < high);
int64_t sqrtx = isqrt(x);
int64_t start = max(y, min(x / high, sqrtx));
int64_t stop = min(x / low, sqrtx);
primesieve::iterator it1(stop, start);
int64_t prime = it1.prev_prime();

if (prime <= start)
return 0;

int threads = 1;
uint64_t xp = (uint64_t)(x / prime);
int64_t pi_xp = pi_noprint(xp, threads);
T sum = pi_xp;
prime = it1.prev_prime();

primesieve::iterator it2(xp + 1, high);
it2.generate_next_primes();

for (; prime > start; prime = it1.prev_prime())
{
xp = (uint64_t)(x / prime);

for (; it2.primes_[it2.size_ - 1] <= xp; it2.generate_next_primes())
pi_xp += it2.size_ - it2.i_;
for (; it2.primes_[it2.i_] <= xp; it2.i_++)
pi_xp += 1;

sum += pi_xp;
}

return sum;
}

template <typename T>
T B_OpenMP(T x,
int64_t y,
int threads,
bool is_print)
{
if (x < 4)
return 0;

T sum = 0;
int64_t xy = (int64_t)(x / max(y, 1));
LoadBalancerP2 loadBalancer(x, xy, threads, is_print);
threads = loadBalancer.get_threads();

#pragma omp parallel num_threads(threads) reduction(+:sum)
{
int64_t low, high;
while (loadBalancer.get_work(low, high))
sum += B_thread(x, y, low, high);
}

return sum;
}

} 

namespace primecount {

int64_t B(int64_t x,
int64_t y,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== B(x, y) ===");
print_gourdon_vars(x, y, threads);
time = get_time();
}

int64_t sum = B_OpenMP((uint64_t) x, y, threads, is_print);

if (is_print)
print("B", sum, time);

return sum;
}

#ifdef HAVE_INT128_T

int128_t B(int128_t x,
int64_t y,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== B(x, y) ===");
print_gourdon_vars(x, y, threads);
time = get_time();
}

int128_t sum = B_OpenMP((uint128_t) x, y, threads, is_print);

if (is_print)
print("B", sum, time);

return sum;
}

#endif

} 
