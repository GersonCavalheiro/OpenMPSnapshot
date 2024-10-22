
#include <primesum-internal.hpp>
#include <aligned_vector.hpp>
#include <BitSieve.hpp>
#include <generate.hpp>
#include <min_max.hpp>
#include <imath.hpp>
#include <PhiTiny.hpp>
#include <S1.hpp>
#include <S2LoadBalancer.hpp>
#include <Wheel.hpp>
#include <int128_t.hpp>
#include <int256_t.hpp>

#include <stdint.h>
#include <vector>

using namespace std;
using namespace primesum;

namespace {

void cross_off(BitSieve& sieve,
int64_t low,
int64_t high,
int64_t prime,
WheelItem& w)
{
int64_t m = w.next_multiple;
int64_t wheel_index = w.wheel_index;

for (; m < high; m += prime * Wheel::next_multiple_factor(&wheel_index))
sieve.unset(m - low);

w.set(m, wheel_index);
}

template <typename T>
T S2_thread(uint128_t x,
int64_t y,
int64_t c,
int64_t segment_size,
int64_t segments_per_thread,
int64_t thread_num,
int64_t low,
int64_t limit,
vector<int32_t>& pi,
vector<int32_t>& primes,
vector<int32_t>& lpf,
vector<int32_t>& mu,
vector<T>& mu_sum,
vector<T>& phi)
{
low += segment_size * segments_per_thread * thread_num;
limit = min(low + segment_size * segments_per_thread, limit);
int64_t size = pi[min(isqrt(x / low), y)] + 1;
int64_t pi_sqrty = pi[isqrt(y)];
int64_t pi_y = pi[y];
T S2_thread = 0;

if (c >= size - 1)
return 0;

BitSieve sieve(segment_size);
Wheel wheel(primes, size, low);
phi.resize(size, 0);
mu_sum.resize(size, 0);

for (; low < limit; low += segment_size)
{
int64_t high = min(low + segment_size, limit);
int64_t b = c + 1;

sieve.pre_sieve(c, low);

for (; b < min(pi_sqrty, size); b++)
{
int64_t prime = primes[b];
int64_t min_m = max(x / (prime * high), y / prime);
int64_t max_m = min(x / (prime * low), y);
int64_t i = 0;

if (prime >= max_m)
goto next_segment;

for (int64_t m = max_m; m > min_m; m--)
{
if (mu[m] != 0 && prime < lpf[m])
{
int64_t xn = x / (prime * m);
int64_t stop = xn - low;
for (; i <= stop; i += 2)
phi[b] += (low + i) * sieve[i];
S2_thread -= phi[b] * (mu[m] * m * prime);
mu_sum[b] -= mu[m] * m * prime;
}
}

for (; i < high - low; i += 2)
phi[b] += (low + i) * sieve[i];

cross_off(sieve, low, high, prime, wheel[b]);
}

for (; b < min(pi_y, size); b++)
{
int64_t prime = primes[b];
int64_t l = pi[min(x / (prime * low), y)];
int64_t min_m = max(x / (prime * high), y / prime, prime);
int64_t i = 0;

if (prime >= primes[l])
goto next_segment;

for (; primes[l] > min_m; l--)
{
int64_t xn = x / (prime * primes[l]);
int64_t stop = xn - low;
for (; i <= stop; i += 2)
phi[b] += (low + i) * sieve[i];
S2_thread += phi[b] * (primes[l] * prime);
mu_sum[b] += primes[l] * prime;
}

for (; i < high - low; i += 2)
phi[b] += (low + i) * sieve[i];

cross_off(sieve, low, high, prime, wheel[b]);
}

next_segment:;
}

return S2_thread;
}

int256_t S2(uint128_t x,
int64_t y,
int64_t c,
vector<int32_t>& primes,
vector<int32_t>& lpf,
vector<int32_t>& mu,
int threads)
{
print("");
print("=== S2(x, y) ===");
print("Computation of the special leaves");

int256_t S2_total = 0;
int64_t low = 1;
int64_t limit = x / y + 1;
threads = ideal_num_threads(threads, limit);

S2LoadBalancer loadBalancer(x, y, limit, threads);
int64_t min_segment_size = loadBalancer.get_min_segment_size();
int64_t segment_size = min_segment_size;
int64_t segments_per_thread = 1;

double time = get_time();
vector<int32_t> pi = generate_pi(y);
vector<int256_t> phi_total(primes.size(), 0);

while (low < limit)
{
int64_t segments = ceil_div(limit - low, segment_size);
threads = in_between(1, threads, segments);
segments_per_thread = in_between(1, segments_per_thread, ceil_div(segments, threads));

aligned_vector<vector<int256_t>> phi(threads);
aligned_vector<vector<int256_t>> mu_sum(threads);
aligned_vector<double> timings(threads);

#pragma omp parallel for num_threads(threads) reduction(+: S2_total)
for (int i = 0; i < threads; i++)
{
timings[i] = get_time();
S2_total += S2_thread(x, y, c, segment_size, segments_per_thread,
i, low, limit, pi, primes, lpf, mu, mu_sum[i], phi[i]);
timings[i] = get_time() - timings[i];
}

for (int i = 0; i < threads; i++)
{
for (size_t j = 1; j < phi[i].size(); j++)
{
S2_total += mu_sum[i][j] * phi_total[j];
phi_total[j] += phi[i][j];
}
}

low += segments_per_thread * threads * segment_size;
loadBalancer.update(low, threads, &segment_size, &segments_per_thread, timings);
}

print("S2", S2_total, time);
return S2_total;
}

} 

namespace primesum {

int256_t pi_lmo_parallel1(int128_t x, int threads)
{
if (x < 2)
return 0;

double alpha = get_alpha_lmo(x);
int64_t x13 = iroot<3>(x);
int64_t y = (int64_t) (x13 * alpha);
int64_t z = x / y;
int64_t c = PhiTiny::get_c(y);

print("");
print("=== pi_lmo_parallel1(x) ===");
print("pi(x) = S1 + S2 + pi(y) - 1 - P2");
print(x, y, z, c, alpha, threads);

int256_t p2 = P2(x, y, threads);
vector<int32_t> mu = generate_moebius(y);
vector<int32_t> lpf = generate_lpf(y);
vector<int32_t> primes = generate_primes(y);

int256_t s1 = S1(x, y, c, threads);
int256_t s2 = S2(x, y, c, primes, lpf, mu, threads);
int256_t phi = s1 + s2;
int256_t sum = phi + prime_sum_tiny(y) - 1 - p2;

return sum;
}

} 
