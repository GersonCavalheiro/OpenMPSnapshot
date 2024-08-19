
#include <primecount-internal.hpp>
#include <Sieve.hpp>
#include <generate.hpp>
#include <generate_phi.hpp>
#include <LoadBalancerS2.hpp>
#include <min.hpp>
#include <imath.hpp>
#include <PhiTiny.hpp>
#include <PiTable.hpp>
#include <print.hpp>
#include <pod_vector.hpp>
#include <S.hpp>

#include <stdint.h>

using namespace primecount;

namespace {

int64_t S2_thread(int64_t x,
int64_t y,
int64_t z,
int64_t c,
const PiTable& pi,
const pod_vector<int32_t>& primes,
const pod_vector<int32_t>& lpf,
const pod_vector<int32_t>& mu,
ThreadData& thread)
{
int64_t sum = 0;
int64_t low = thread.low;
int64_t low1 = max(low, 1);
int64_t segments = thread.segments;
int64_t segment_size = thread.segment_size;
int64_t pi_sqrty = pi[isqrt(y)];
int64_t limit = min(low + segments * segment_size, z + 1);
int64_t max_b = pi[min(isqrt(x / low1), y - 1)];
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

for (; b <= min(pi_sqrty, max_b); b++)
{
int64_t prime = primes[b];
int64_t min_m = max(x / (prime * high), y / prime);
int64_t max_m = min(x / (prime * low1), y);

if (prime >= max_m)
goto next_segment;

for (int64_t m = max_m; m > min_m; m--)
{
if (mu[m] != 0 && prime < lpf[m])
{
int64_t xpm = x / (prime * m);
int64_t stop = xpm - low;
int64_t phi_xpm = phi[b] + sieve.count(stop);
sum -= mu[m] * phi_xpm;
}
}

phi[b] += sieve.get_total_count();
sieve.cross_off_count(prime, b);
}

for (; b <= max_b; b++)
{
int64_t prime = primes[b];
int64_t l = pi[min(x / (prime * low1), y)];
int64_t min_m = max(x / (prime * high), prime);

if (prime >= primes[l])
goto next_segment;

for (; primes[l] > min_m; l--)
{
int64_t xpq = x / (prime * primes[l]);
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

int64_t S2(int64_t x,
int64_t y,
int64_t z,
int64_t c,
int64_t s2_approx,
const pod_vector<int32_t>& primes,
const pod_vector<int32_t>& lpf,
const pod_vector<int32_t>& mu,
int threads,
bool is_print)
{
double time;

if (is_print)
{
print("");
print("=== S2(x, y) ===");
time = get_time();
}

int64_t thread_threshold = 1 << 20;
int max_threads = (int) std::pow(z, 1 / 3.7);
threads = std::min(threads, max_threads);
threads = ideal_num_threads(z, threads, thread_threshold);
LoadBalancerS2 loadBalancer(x, z, s2_approx, threads, is_print);
PiTable pi(y, threads);

#pragma omp parallel num_threads(threads)
{
ThreadData thread;

while (loadBalancer.get_work(thread))
{
thread.start_time();
thread.sum = S2_thread(x, y, z, c, pi, primes, lpf, mu, thread);
thread.stop_time();
}
}

int64_t sum = (int64_t) loadBalancer.get_sum();

if (is_print)
print("S2", sum, time);

return sum;
}

} 

namespace primecount {

int64_t pi_lmo_parallel(int64_t x,
int threads,
bool is_print)
{
if (x < 2)
return 0;

double alpha = get_alpha_lmo(x);
int64_t x13 = iroot<3>(x);
int64_t y = (int64_t) (x13 * alpha);
int64_t z = x / y;
int64_t c = PhiTiny::get_c(y);

if (is_print)
{
print("");
print("=== pi_lmo_parallel(x) ===");
print("pi(x) = S1 + S2 + pi(y) - 1 - P2");
print(x, y, z, c, threads);
}

auto primes = generate_primes<int32_t>(y);
auto lpf = generate_lpf(y);
auto mu = generate_moebius(y);

int64_t pi_y = primes.size() - 1;
int64_t p2 = P2(x, y, pi_y, threads, is_print);
int64_t s1 = S1(x, y, c, threads, is_print);
int64_t s2_approx = S2_approx(x, pi_y, p2, s1);
int64_t s2 = S2(x, y, z, c, s2_approx, primes, lpf, mu, threads, is_print);
int64_t phi = s1 + s2;
int64_t sum = phi + pi_y - 1 - p2;

return sum;
}

} 
