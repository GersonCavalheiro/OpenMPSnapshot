
#include <PiTable.hpp>
#include <primesum-internal.hpp>
#include <generate.hpp>
#include <imath.hpp>
#include <PhiTiny.hpp>
#include <fast_div.hpp>
#include <min_max.hpp>

#include <stdint.h>
#include <algorithm>
#include <array>
#include <vector>
#include <limits>

using namespace std;
using namespace primesum;

namespace {

const int MAX_A = 100;

class PhiCache
{
public:
PhiCache(vector<int32_t>& primes,
PiTable& pi) :
primes_(primes),
pi_(pi)
{ }

template <int SIGN>
int64_t phi(int64_t x, int64_t a)
{
if (x <= primes_[a])
return SIGN;
else if (is_phi_tiny(a))
return phi_tiny(x, a) * SIGN;
else if (is_pix(x, a))
return (pi_[x] - a + 1) * SIGN;
else if (is_cached(x, a))
return cache_[a][x] * SIGN;

int64_t sqrtx = isqrt(x);
int64_t pi_sqrtx = a;
int64_t c = PhiTiny::get_c(sqrtx);
int64_t sum = 0;

if (sqrtx < pi_.size())
pi_sqrtx = min(pi_[sqrtx], a);

sum += (pi_sqrtx - a) * SIGN;
sum += phi_tiny(x, c) * SIGN;

for (int64_t i = c; i < pi_sqrtx; i++)
{
int64_t x2 = fast_div(x, primes_[i + 1]);

if (is_pix(x2, i))
sum += (pi_[x2] - i + 1) * -SIGN;
else
sum += phi<-SIGN>(x2, i);
}

update_cache(x, a, sum);

return sum;
}

private:
using T = uint16_t;
array<vector<T>, MAX_A> cache_;
vector<int32_t>& primes_;
PiTable& pi_;

void update_cache(uint64_t x, uint64_t a, int64_t sum)
{
if (a < cache_.size() &&
x <= numeric_limits<T>::max())
{
if (x >= cache_[a].size())
cache_[a].resize(x + 1, 0);

cache_[a][x] = (T) abs(sum);
}
}

bool is_pix(int64_t x, int64_t a) const
{
return x < pi_.size() &&
x < isquare(primes_[a + 1]);
}

bool is_cached(uint64_t x, uint64_t a) const
{
return a < cache_.size() && 
x < cache_[a].size() && 
cache_[a][x];
}
};

} 

namespace primesum {

int64_t phi(int64_t x, int64_t a, int threads)
{
if (x < 1) return 0;
if (a > x) return 1;
if (a < 1) return x;

int64_t sum = 0;

if (is_phi_tiny(a))
sum = phi_tiny(x, a);
else
{
auto primes = generate_n_primes(a);

if (primes[a] >= x)
sum = 1;
else
{
int64_t sqrtx = isqrt(x);
PiTable pi(max(sqrtx, primes[a]));
PhiCache cache(primes, pi);

int64_t c = PhiTiny::get_c(sqrtx);
int64_t pi_sqrtx = min(pi[sqrtx], a);
int64_t thread_threshold = ipow(10ll, 10);
threads = ideal_num_threads(threads, x, thread_threshold);

sum = phi_tiny(x, c) - a + pi_sqrtx;

#pragma omp parallel for num_threads(threads) schedule(dynamic, 16) firstprivate(cache) reduction(+: sum)
for (int64_t i = c; i < pi_sqrtx; i++)
sum += cache.phi<-1>(x / primes[i + 1], i);
}
}

return sum;
}

} 
