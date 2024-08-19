
#include <primecount-internal.hpp>
#include <PiTable.hpp>
#include <FactorTable.hpp>
#include <Sieve.hpp>
#include <fast_div.hpp>
#include <generate.hpp>
#include <generate_phi.hpp>
#include <imath.hpp>
#include <int128_t.hpp>
#include <LoadBalancerS2.hpp>
#include <min.hpp>
#include <print.hpp>
#include <S.hpp>

#include <stdint.h>

using namespace primecount;

namespace {

template <typename T, typename Primes, typename FactorTable>
T S2_hard_thread(T x,
int64_t y,
int64_t z,
int64_t c,
const Primes& primes,
const PiTable& pi,
const FactorTable& factor,
ThreadData& thread)
{
T sum = 0;

int64_t low = thread.low;
int64_t low1 = max(low, 1);
int64_t segments = thread.segments;
int64_t segment_size = thread.segment_size;
int64_t limit = min(low + segments * segment_size, z);
int64_t pi_sqrty = pi[isqrt(y)];
int64_t max_b = (limit <= y) ? pi_sqrty
: pi[min3(isqrt(x / low1), isqrt(z), y)];
int64_t min_b = pi[min(z / limit, primes[max_b])];
min_b = max(c, min_b) + 1;

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

for (int64_t last = min(pi_sqrty, max_b); b <= last; b++)
{
int64_t prime = primes[b];
T xp = x / prime;
int64_t xp_high = min(fast_div(xp, high), y);
int64_t min_m = max(xp_high, y / prime);
int64_t max_m = min(fast_div(xp, low1), y);

if (prime >= max_m)
goto next_segment;

min_m = factor.to_index(min_m);
max_m = factor.to_index(max_m);

for (int64_t m = max_m; m > min_m; m--)
{
if (prime < factor.mu_lpf(m))
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
int64_t l = pi[min(xp_low, z / prime)];
int64_t min_hard = max(xp_high, prime);

if (prime >= primes[l])
goto next_segment;

for (; primes[l] > min_hard; l--)
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

template <typename T, typename Primes, typename FactorTable>
T S2_hard_OpenMP(T x,
int64_t y,
int64_t z,
int64_t c,
T s2_hard_approx,
const Primes& primes,
const FactorTable& factor,
int threads,
bool is_print)
{
int64_t thread_threshold = 1 << 20;
int max_threads = (int) std::pow(z, 1 / 3.7);
threads = std::min(threads, max_threads);
threads = ideal_num_threads(z, threads, thread_threshold);

LoadBalancerS2 loadBalancer(x, z, s2_hard_approx, threads, is_print);
int64_t max_prime = min(y, z / isqrt(y));
PiTable pi(max_prime, threads);

#pragma omp parallel num_threads(threads)
{
ThreadData thread;

while (loadBalancer.get_work(thread))
{
using UT = typename std::make_unsigned<T>::type;

thread.start_time();
UT sum = S2_hard_thread((UT) x, y, z, c, primes, pi, factor, thread);
thread.sum = (T) sum;
thread.stop_time();
}
}

T sum = (T) loadBalancer.get_sum();

return sum;
}

} 

namespace primecount {

int64_t S2_hard(int64_t x,
int64_t y,
int64_t z,
int64_t c,
int64_t s2_hard_approx,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== S2_hard(x, y) ===");
print_vars(x, y, c, threads);
time = get_time();
}

FactorTable<uint16_t> factor(y, threads);
int64_t max_prime = min(y, z / isqrt(y));
auto primes = generate_primes<int32_t>(max_prime);
int64_t sum = S2_hard_OpenMP(x, y, z, c, s2_hard_approx, primes, factor, threads, is_print);

if (is_print)
print("S2_hard", sum, time);

return sum;
}

#ifdef HAVE_INT128_T

int128_t S2_hard(int128_t x,
int64_t y,
int64_t z,
int64_t c,
int128_t s2_hard_approx,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== S2_hard(x, y) ===");
print_vars(x, y, c, threads);
time = get_time();
}

int128_t sum;

if (y <= FactorTable<uint16_t>::max())
{
FactorTable<uint16_t> factor(y, threads);
int64_t max_prime = min(y, z / isqrt(y));
auto primes = generate_primes<uint32_t>(max_prime);
sum = S2_hard_OpenMP(x, y, z, c, s2_hard_approx, primes, factor, threads, is_print);
}
else
{
FactorTable<uint32_t> factor(y, threads);
int64_t max_prime = min(y, z / isqrt(y));
auto primes = generate_primes<int64_t>(max_prime);
sum = S2_hard_OpenMP(x, y, z, c, s2_hard_approx, primes, factor, threads, is_print);
}

if (is_print)
print("S2_hard", sum, time);

return sum;
}

#endif

} 
