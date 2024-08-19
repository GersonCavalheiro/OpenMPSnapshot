
#include <primecount-internal.hpp>
#include <FactorTableD.hpp>
#include <PiTable.hpp>
#include <Sieve.hpp>
#include <LoadBalancerS2.hpp>
#include <fast_div.hpp>
#include <generate.hpp>
#include <generate_phi.hpp>
#include <gourdon.hpp>
#include <imath.hpp>
#include <int128_t.hpp>
#include <min.hpp>
#include <print.hpp>

#include <stdint.h>

using namespace primecount;

namespace {

template <typename T, typename Primes, typename FactorTableD>
T D_thread(T x,
int64_t x_star,
int64_t xz,
int64_t y,
int64_t z,
int64_t k,
const Primes& primes,
const PiTable& pi,
const FactorTableD& factor,
ThreadData& thread)
{
T sum = 0;

int64_t low = thread.low;
int64_t low1 = max(low, 1);
int64_t segments = thread.segments;
int64_t segment_size = thread.segment_size;
int64_t pi_sqrtz = pi[isqrt(z)];
int64_t limit = min(low + segments * segment_size, xz);
int64_t max_b = pi[min3(isqrt(x / low1), isqrt(limit), x_star)];
int64_t min_b = pi[min(xz / limit, x_star)];
min_b = max(k, min_b) + 1;

if (min_b > max_b)
return 0;

auto phi = generate_phi(low, max_b, primes, pi);
Sieve sieve(low, segment_size, max_b);
thread.init_finished();

for (; low < limit; low += segment_size)
{
int64_t high = min(low + segment_size, limit);
low1 = max(low, 1);

sieve.pre_sieve(primes, min_b - 1, low, high);
int64_t b = min_b;

for (int64_t last = min(pi_sqrtz, max_b); b <= last; b++)
{
int64_t prime = primes[b];
T xp = x / prime;
int64_t xp_low = min(fast_div(xp, low1), z);
int64_t xp_high = min(fast_div(xp, high), z);
int64_t min_m = max(xp_high, z / prime);
int64_t max_m = min(fast_div(xp, prime * prime), xp_low);

if (prime >= max_m)
goto next_segment;

min_m = factor.to_index(min_m);
max_m = factor.to_index(max_m);

for (int64_t m = max_m; m > min_m; m--)
{
if (prime < factor.is_leaf(m))
{
int64_t xpm = fast_div64(xp, factor.to_number(m));
int64_t stop = xpm - low;
int64_t phi_xpm = phi[b] + sieve.count(stop);
int64_t mu_m = factor.mu(m);
sum -= mu_m * phi_xpm;
}
}

phi[b] += sieve.get_total_count();
sieve.cross_off_count(prime, b);
}

for (; b <= max_b; b++)
{
int64_t prime = primes[b];
T xp = x / prime;
int64_t xp_low = min(fast_div(xp, low1), y);
int64_t xp_high = min(fast_div(xp, high), y);
int64_t min_m = max(xp_high, prime);
int64_t max_m = min(fast_div(xp, prime * prime), xp_low);
int64_t l = pi[max_m];

if (prime >= primes[l])
goto next_segment;

for (; primes[l] > min_m; l--)
{
int64_t xpq = fast_div64(xp, primes[l]);
int64_t stop = xpq - low;
int64_t phi_xpq = phi[b] + sieve.count(stop);
sum += phi_xpq;
}

phi[b] += sieve.get_total_count();
sieve.cross_off_count(prime, b);
}

next_segment:;
}

return sum;
}

template <typename T, typename Primes, typename FactorTableD>
T D_OpenMP(T x,
int64_t y,
int64_t z,
int64_t k,
T d_approx,
const Primes& primes,
const FactorTableD& factor,
int threads,
bool is_print)
{
int64_t xz = x / z;
int64_t x_star = get_x_star_gourdon(x, y);

int64_t thread_threshold = 1 << 20;
int max_threads = (int) std::pow(xz, 1 / 3.7);
threads = std::min(threads, max_threads);
threads = ideal_num_threads(xz, threads, thread_threshold);
LoadBalancerS2 loadBalancer(x, xz, d_approx, threads, is_print);
PiTable pi(y, threads);

#pragma omp parallel num_threads(threads)
{
ThreadData thread;

while (loadBalancer.get_work(thread))
{
using UT = typename std::make_unsigned<T>::type;

thread.start_time();
UT sum = D_thread((UT) x, x_star, xz, y, z, k, primes, pi, factor, thread);
thread.sum = (T) sum;
thread.stop_time();
}
}

T sum = (T) loadBalancer.get_sum();

return sum;
}

} 

namespace primecount {

int64_t D(int64_t x,
int64_t y,
int64_t z,
int64_t k,
int64_t d_approx,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== D(x, y) ===");
print_gourdon_vars(x, y, z, k, threads);
time = get_time();
}

FactorTableD<uint16_t> factor(y, z, threads);
auto primes = generate_primes<int32_t>(y);
int64_t sum = D_OpenMP(x, y, z, k, d_approx, primes, factor, threads, is_print);

if (is_print)
print("D", sum, time);

return sum;
}

#ifdef HAVE_INT128_T

int128_t D(int128_t x,
int64_t y,
int64_t z,
int64_t k,
int128_t d_approx,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== D(x, y) ===");
print_gourdon_vars(x, y, z, k, threads);
time = get_time();
}

int128_t sum;

if (z <= FactorTableD<uint16_t>::max())
{
FactorTableD<uint16_t> factor(y, z, threads);
auto primes = generate_primes<uint32_t>(y);
sum = D_OpenMP(x, y, z, k, d_approx, primes, factor, threads, is_print);
}
else
{
FactorTableD<uint32_t> factor(y, z, threads);
auto primes = generate_primes<int64_t>(y);
sum = D_OpenMP(x, y, z, k, d_approx, primes, factor, threads, is_print);
}

if (is_print)
print("D", sum, time);

return sum;
}

#endif

} 
