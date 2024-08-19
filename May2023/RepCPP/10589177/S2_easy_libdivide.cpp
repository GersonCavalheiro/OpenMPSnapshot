
#include <PiTable.hpp>
#include <primecount-internal.hpp>
#include <fast_div.hpp>
#include <generate.hpp>
#include <int128_t.hpp>
#include <min.hpp>
#include <imath.hpp>
#include <pod_vector.hpp>
#include <print.hpp>
#include <RelaxedAtomic.hpp>
#include <StatusS2.hpp>
#include <S.hpp>

#include <libdivide.h>
#include <stdint.h>

using std::numeric_limits;
using namespace primecount;

namespace {

template <typename T,
typename LibdividePrimes>
T S2_easy_64(T xp128,
uint64_t y,
uint64_t z,
uint64_t b,
uint64_t prime,
const LibdividePrimes& primes,
const PiTable& pi)
{
uint64_t xp = (uint64_t) xp128;
uint64_t min_trivial = min(xp / prime, y);
uint64_t min_clustered = isqrt(xp);
uint64_t min_sparse = z / prime;
min_clustered = in_between(prime, min_clustered, y);
min_sparse = in_between(prime, min_sparse, y);
uint64_t l = pi[min_trivial];
uint64_t pi_min_clustered = pi[min_clustered];
uint64_t pi_min_sparse = pi[min_sparse];

T sum = 0;

while (l > pi_min_clustered)
{
uint64_t xpq = xp / primes[l];
uint64_t pi_xpq = pi[xpq];
uint64_t phi_xpq = pi_xpq - b + 2;
uint64_t xpq2 = xp / primes[pi_xpq + 1];
uint64_t lmin = pi[xpq2];
sum += phi_xpq * (l - lmin);
l = lmin;
}

for (; l > pi_min_sparse; l--)
{
uint64_t xpq = xp / primes[l];
sum += pi[xpq] - b + 2;
}

return sum;
}

template <typename T,
typename Primes>
T S2_easy_128(T xp,
uint64_t y,
uint64_t z,
uint64_t b,
uint64_t prime,
const Primes& primes,
const PiTable& pi)
{
uint64_t min_trivial = min(xp / prime, y);
uint64_t min_clustered = (uint64_t) isqrt(xp);
uint64_t min_sparse = z / prime;
min_clustered = in_between(prime, min_clustered, y);
min_sparse = in_between(prime, min_sparse, y);
uint64_t l = pi[min_trivial];
uint64_t pi_min_clustered = pi[min_clustered];
uint64_t pi_min_sparse = pi[min_sparse];

T sum = 0;

while (l > pi_min_clustered)
{
uint64_t xpq = fast_div64(xp, primes[l]);
uint64_t phi_xpq = pi[xpq] - b + 2;
uint64_t xpq2 = fast_div64(xp, primes[b + phi_xpq - 1]);
uint64_t lmin = pi[xpq2];
sum += phi_xpq * (l - lmin);
l = lmin;
}

for (; l > pi_min_sparse; l--)
{
uint64_t xpq = fast_div64(xp, primes[l]);
sum += pi[xpq] - b + 2;
}

return sum;
}

template <typename T,
typename Primes>
T S2_easy_OpenMP(T x,
int64_t y,
int64_t z,
int64_t c,
const Primes& primes,
int threads,
bool is_print)
{
pod_vector<libdivide::branchfree_divider<uint64_t>> lprimes;
lprimes.resize(primes.size());
for (std::size_t i = 1; i < lprimes.size(); i++)
lprimes[i] = primes[i];

T sum = 0;
int64_t x13 = iroot<3>(x); 

int64_t thread_threshold = 1000;
int max_threads = (int) std::pow(z, 1 / 4.0);
threads = std::min(threads, max_threads);
threads = ideal_num_threads(x13, threads, thread_threshold);

StatusS2 status(x);
PiTable pi(y, threads);
int64_t pi_sqrty = pi[isqrt(y)];
int64_t pi_x13 = pi[x13];
RelaxedAtomic<int64_t> min_b(max(c, pi_sqrty) + 1);

#pragma omp parallel num_threads(threads) reduction(+: sum)
for (int64_t b = min_b++; b <= pi_x13; b = min_b++)
{
int64_t prime = primes[b];
T xp = x / prime;

if (xp <= numeric_limits<uint64_t>::max())
sum += S2_easy_64(xp, y, z, b, prime, lprimes, pi);
else
sum += S2_easy_128(xp, y, z, b, prime, primes, pi);

#pragma omp master
if (is_print)
status.print(b, pi_x13);
}

return sum;
}

} 

namespace primecount {

int64_t S2_easy(int64_t x,
int64_t y,
int64_t z,
int64_t c,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== S2_easy(x, y) ===");
print_vars(x, y, c, threads);
time = get_time();
}

auto primes = generate_primes<uint32_t>(y);
int64_t sum = S2_easy_OpenMP((uint64_t) x, y, z, c, primes, threads, is_print);

if (is_print)
print("S2_easy", sum, time);

return sum;
}

#ifdef HAVE_INT128_T

int128_t S2_easy(int128_t x,
int64_t y,
int64_t z,
int64_t c,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== S2_easy(x, y) ===");
print_vars(x, y, c, threads);
time = get_time();
}

int128_t sum;

if (y <= numeric_limits<uint32_t>::max())
{
auto primes = generate_primes<uint32_t>(y);
sum = S2_easy_OpenMP((uint128_t) x, y, z, c, primes, threads, is_print);
}
else
{
auto primes = generate_primes<int64_t>(y);
sum = S2_easy_OpenMP((uint128_t) x, y, z, c, primes, threads, is_print);
}

if (is_print)
print("S2_easy", sum, time);

return sum;
}

#endif

} 
