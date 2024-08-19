
#include <gourdon.hpp>
#include <primecount-internal.hpp>
#include <PhiTiny.hpp>
#include <generate.hpp>
#include <imath.hpp>
#include <int128_t.hpp>
#include <print.hpp>
#include <pod_vector.hpp>

#include <stdint.h>

using std::numeric_limits;
using namespace primecount;

namespace {

template <int MU, typename T, typename P>
T Phi0_thread(T x,
int64_t z,
uint64_t b,
int64_t k,
T square_free,
const pod_vector<P>& primes)
{
T phi0 = 0;

for (b++; b < primes.size(); b++)
{
T next = square_free * primes[b];
if (next > z) break;
phi0 += MU * phi_tiny(x / next, k);
phi0 += Phi0_thread<-MU>(x, z, b, k, next, primes);
}

return phi0;
}

template <typename X, typename Y>
X Phi0_OpenMP(X x,
Y y,
int64_t z,
int64_t k,
int threads)
{
int64_t thread_threshold = (int64_t) 1e6;
threads = ideal_num_threads(y, threads, thread_threshold);

auto primes = generate_primes<Y>(y);
int64_t pi_y = primes.size() - 1;
X phi0 = phi_tiny(x, k);

#pragma omp parallel for schedule(static, 1) num_threads(threads) reduction (+: phi0)
for (int64_t b = k + 1; b <= pi_y; b++)
{
phi0 -= phi_tiny(x / primes[b], k);
phi0 += Phi0_thread<1>(x, z, b, k, (X) primes[b], primes);
}

return phi0;
}

} 

namespace primecount {

int64_t Phi0(int64_t x,
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
print("=== Phi0(x, y) ===");
print_gourdon_vars(x, y, z, k, threads);
time = get_time();
}

int64_t phi0 = Phi0_OpenMP(x, y, z, k, threads);

if (is_print)
print("Phi0", phi0, time);

return phi0;
}

#ifdef HAVE_INT128_T

int128_t Phi0(int128_t x,
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
print("=== Phi0(x, y) ===");
print_gourdon_vars(x, y, z, k, threads);
time = get_time();
}

int128_t phi0;

if (y <= numeric_limits<uint32_t>::max())
phi0 = Phi0_OpenMP(x, (uint32_t) y, z, k, threads);
else
phi0 = Phi0_OpenMP(x, y, z, k, threads);

if (is_print)
print("Phi0", phi0, time);

return phi0;
}

#endif

} 
