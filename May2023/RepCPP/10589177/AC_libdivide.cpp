
#include <PiTable.hpp>
#include <SegmentedPiTable.hpp>
#include <primecount-internal.hpp>
#include <LoadBalancerAC.hpp>
#include <fast_div.hpp>
#include <generate.hpp>
#include <gourdon.hpp>
#include <int128_t.hpp>
#include <libdivide.h>
#include <min.hpp>
#include <imath.hpp>
#include <pod_vector.hpp>
#include <print.hpp>
#include <RelaxedAtomic.hpp>

#include <stdint.h>

using std::numeric_limits;
using namespace primecount;

namespace {

template <typename T,
typename LibdividePrimes>
T A_64(T xlow,
T xhigh,
uint64_t xp,
uint64_t y,
uint64_t prime,
const LibdividePrimes& primes,
const PiTable& pi,
const SegmentedPiTable& segmentedPi)
{
T sum = 0;

uint64_t sqrt_xp = isqrt(xp);
uint64_t min_2nd_prime = min(xhigh / prime, sqrt_xp);
uint64_t max_2nd_prime = min(xlow / prime, sqrt_xp);
uint64_t i = pi[max(prime, min_2nd_prime)] + 1;
uint64_t max_i1 = pi[min(xp / y, max_2nd_prime)];
uint64_t max_i2 = pi[max_2nd_prime];

for (; i <= max_i1; i++)
{
uint64_t xpq = xp / primes[i];
sum += segmentedPi[xpq];
}

for (; i <= max_i2; i++)
{
uint64_t xpq = xp / primes[i];
sum += segmentedPi[xpq] * 2;
}

return sum;
}

template <typename T,
typename Primes>
T A_128(T xlow,
T xhigh,
T xp,
uint64_t y,
uint64_t prime,
const Primes& primes,
const PiTable& pi,
const SegmentedPiTable& segmentedPi)
{
T sum = 0;

uint64_t sqrt_xp = (uint64_t) isqrt(xp);
uint64_t min_2nd_prime = min(xhigh / prime, sqrt_xp);
uint64_t max_2nd_prime = min(xlow / prime, sqrt_xp);
uint64_t i = pi[max(prime, min_2nd_prime)] + 1;
uint64_t max_i1 = pi[min(xp / y, max_2nd_prime)];
uint64_t max_i2 = pi[max_2nd_prime];

for (; i <= max_i1; i++)
{
uint64_t xpq = fast_div64(xp, primes[i]);
sum += segmentedPi[xpq];
}

for (; i <= max_i2; i++)
{
uint64_t xpq = fast_div64(xp, primes[i]);
sum += segmentedPi[xpq] * 2;
}

return sum;
}

template <int MU, 
typename T, 
typename Primes>
T C1(T xp,
uint64_t b,
uint64_t i,
uint64_t pi_y,
uint64_t m,
uint64_t min_m,
uint64_t max_m,
const Primes& primes,
const PiTable& pi)
{
T sum = 0;

for (i++; i <= pi_y; i++)
{
T m128 = (T) m * primes[i];
if (m128 > max_m)
return sum;

uint64_t m64 = (uint64_t) m128;

if (m64 > min_m) {
uint64_t xpm = fast_div64(xp, m64);
T phi_xpm = pi[xpm] - b + 2;
sum += phi_xpm * MU;
}

sum += C1<-MU>(xp, b, i, pi_y, m64, min_m, max_m, primes, pi);
}

return sum;
}

template <typename T, 
typename LibdividePrimes>
T C2_64(T xlow,
T xhigh,
uint64_t xp,
uint64_t y,
uint64_t b,
uint64_t prime,
const LibdividePrimes& primes,
const PiTable& pi,
const SegmentedPiTable& segmentedPi)
{
T sum = 0;

uint64_t max_m = min3(xlow / prime, xp / prime, y);
T min_m128 = max3(xhigh / prime, xp / (prime * prime), prime);
uint64_t min_m = min(min_m128, max_m);
uint64_t i = pi[max_m];
uint64_t pi_min_m = pi[min_m];
uint64_t min_clustered = isqrt(xp);
min_clustered = in_between(min_m, min_clustered, max_m);
uint64_t pi_min_clustered = pi[min_clustered];

while (i > pi_min_clustered)
{
uint64_t xpq = xp / primes[i];
uint64_t pi_xpq = segmentedPi[xpq];
uint64_t phi_xpq = pi_xpq - b + 2;
uint64_t xpq2 = xp / primes[pi_xpq + 1];
uint64_t imin = pi[max(xpq2, min_clustered)];
sum += phi_xpq * (i - imin);
i = imin;
}

for (; i > pi_min_m; i--)
{
uint64_t xpq = xp / primes[i];
sum += segmentedPi[xpq] - b + 2;
}

return sum;
}

template <typename T,
typename Primes>
T C2_128(T xlow,
T xhigh,
T xp,
uint64_t y,
uint64_t b,
const Primes& primes,
const PiTable& pi,
const SegmentedPiTable& segmentedPi)
{
T sum = 0;

uint64_t prime = primes[b];
uint64_t max_m = min3(xlow / prime, xp / prime, y);
T min_m128 = max3(xhigh / prime, xp / (prime * prime), prime);
uint64_t min_m = min(min_m128, max_m);
uint64_t i = pi[max_m];
uint64_t pi_min_m = pi[min_m];
uint64_t min_clustered = (uint64_t) isqrt(xp);
min_clustered = in_between(min_m, min_clustered, max_m);
uint64_t pi_min_clustered = pi[min_clustered];

while (i > pi_min_clustered)
{
uint64_t xpq = fast_div64(xp, primes[i]);
uint64_t pi_xpq = segmentedPi[xpq];
uint64_t phi_xpq = pi_xpq - b + 2;
uint64_t xpq2 = fast_div64(xp, primes[pi_xpq + 1]);
uint64_t imin = pi[max(xpq2, min_clustered)];
sum += phi_xpq * (i - imin);
i = imin;
}

for (; i > pi_min_m; i--)
{
uint64_t xpq = fast_div64(xp, primes[i]);
sum += segmentedPi[xpq] - b + 2;
}

return sum;
}

template <typename T,
typename Primes>
T AC_OpenMP(T x,
int64_t y,
int64_t z,
int64_t k,
int64_t x_star,
int64_t max_a_prime,
const Primes& primes,
int threads,
bool is_print)
{
T sum = 0;
int64_t x13 = iroot<3>(x);
int64_t sqrtx = isqrt(x);
int64_t xy = x / y;
int64_t xz = x / z;

int64_t thread_threshold = 1000;
int max_threads = (int) std::pow(xz, 1 / 3.7);
threads = std::min(threads, max_threads);
threads = ideal_num_threads(x13, threads, thread_threshold);
LoadBalancerAC loadBalancer(sqrtx, y, threads, is_print);

pod_vector<libdivide::branchfree_divider<uint64_t>> lprimes;
lprimes.resize(primes.size());
for (std::size_t i = 1; i < lprimes.size(); i++)
lprimes[i] = primes[i];

PiTable pi(max(z, max_a_prime), threads);

int64_t pi_y = pi[y];
int64_t pi_sqrtz = pi[isqrt(z)];
int64_t pi_root3_xy = pi[iroot<3>(xy)];
int64_t pi_root3_xz = pi[iroot<3>(xz)];
RelaxedAtomic<int64_t> min_c1(max(k, pi_root3_xz) + 1);

#pragma omp parallel num_threads(threads) reduction(+: sum)
{
SegmentedPiTable segmentedPi;
int64_t low, high;

for (int64_t b = min_c1++; b <= pi_sqrtz; b = min_c1++)
{
int64_t prime = primes[b];
T xp = x / prime;
int64_t max_m = min(xp / prime, z);
T min_m128 = max(xp / (prime * prime), z / prime);
int64_t min_m = min(min_m128, max_m);

sum -= C1<-1>(xp, b, b, pi_y, 1, min_m, max_m, primes, pi);
}

while (loadBalancer.get_work(low, high))
{
segmentedPi.init(low, high);
T xlow = x / max(low, 1);
T xhigh = x / high;

int64_t min_c2 = max(k, pi_root3_xy);
min_c2 = max(min_c2, pi_sqrtz);
min_c2 = max(min_c2, pi[isqrt(low)]);
min_c2 = max(min_c2, pi[min(xhigh / y, x_star)]);
min_c2 += 1;

int64_t min_a = min(xhigh / high, x13);
min_a = pi[max(x_star, min_a)] + 1;

T sqrt_xlow = isqrt(xlow);
int64_t max_c2 = pi[min(sqrt_xlow, x_star)];
int64_t max_a = pi[min(sqrt_xlow, x13)];

for (int64_t b = min_c2; b <= max_c2; b++)
{
int64_t prime = primes[b];
T xp = x / prime;

if (xp <= numeric_limits<uint64_t>::max())
sum += C2_64(xlow, xhigh, (uint64_t) xp, y, b, prime, lprimes, pi, segmentedPi);
else
sum += C2_128(xlow, xhigh, xp, y, b, primes, pi, segmentedPi);
}

for (int64_t b = min_a; b <= max_a; b++)
{
int64_t prime = primes[b];
T xp = x / prime;

if (xp <= numeric_limits<uint64_t>::max())
sum += A_64(xlow, xhigh, (uint64_t) xp, y, prime, lprimes, pi, segmentedPi);
else
sum += A_128(xlow, xhigh, xp, y, prime, primes, pi, segmentedPi);
}
}
}

return sum;
}

} 

namespace primecount {

int64_t AC(int64_t x,
int64_t y,
int64_t z,
int64_t k,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== AC(x, y) ===");
print_gourdon_vars(x, y, z, k, threads);
time = get_time();
}

int64_t x_star = get_x_star_gourdon(x, y);
int64_t max_c_prime = y;
int64_t max_a_prime = (int64_t) isqrt(x / x_star);
int64_t max_prime = max(max_a_prime, max_c_prime);
auto primes = generate_primes<uint32_t>(max_prime);

int64_t sum = AC_OpenMP((uint64_t) x, y, z, k, x_star, max_a_prime, primes, threads, is_print);

if (is_print)
print("A + C", sum, time);

return sum;
}

#ifdef HAVE_INT128_T

int128_t AC(int128_t x,
int64_t y,
int64_t z,
int64_t k,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== AC(x, y) ===");
print_gourdon_vars(x, y, z, k, threads);
time = get_time();
}

int64_t x_star = get_x_star_gourdon(x, y);
int64_t max_c_prime = y;
int64_t max_a_prime = (int64_t) isqrt(x / x_star);
int64_t max_prime = max(max_a_prime, max_c_prime);
int128_t sum;

if (max_prime <= numeric_limits<uint32_t>::max())
{
auto primes = generate_primes<uint32_t>(max_prime);
sum = AC_OpenMP((uint128_t) x, y, z, k, x_star, max_a_prime, primes, threads, is_print);
}
else
{
auto primes = generate_primes<uint64_t>(max_prime);
sum = AC_OpenMP((uint128_t) x, y, z, k, x_star, max_a_prime, primes, threads, is_print);
}

if (is_print)
print("A + C", sum, time);

return sum;
}

#endif

} 
