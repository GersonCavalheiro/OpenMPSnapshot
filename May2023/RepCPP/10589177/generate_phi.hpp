
#ifndef GENERATE_PHI_HPP
#define GENERATE_PHI_HPP

#include <primecount-internal.hpp>
#include <BitSieve240.hpp>
#include <fast_div.hpp>
#include <imath.hpp>
#include <macros.hpp>
#include <min.hpp>
#include <PhiTiny.hpp>
#include <PiTable.hpp>
#include <pod_vector.hpp>
#include <popcnt.hpp>

#include <stdint.h>
#include <algorithm>
#include <utility>

namespace {

using namespace primecount;

template <typename Primes>
class PhiCache : public BitSieve240
{
public:
PhiCache(uint64_t x,
uint64_t a,
const Primes& primes,
const PiTable& pi) :
primes_(primes),
pi_(pi)
{
uint64_t max_a = 100;

a = a - min(a, 30);
max_a = min(a, max_a);

if (max_a <= PhiTiny::max_a())
return;

uint64_t max_x = isqrt(x);

uint64_t max_megabytes = 16;
uint64_t indexes = max_a - PhiTiny::max_a();
uint64_t max_bytes = max_megabytes << 20;
uint64_t max_bytes_per_index = max_bytes / indexes;
uint64_t numbers_per_byte = 240 / sizeof(sieve_t);
uint64_t cache_limit = max_bytes_per_index * numbers_per_byte;
max_x = min(max_x, cache_limit);
max_x_size_ = ceil_div(max_x, 240);

if (max_x_size_ < 8)
return;

max_x_ = max_x_size_ * 240 - 1;
max_a_ = max_a;
}

template <int SIGN>
int64_t phi(int64_t x, int64_t a)
{
if (x <= (int64_t) primes_[a])
return SIGN;
else if (is_phi_tiny(a))
return phi_tiny(x, a) * SIGN;
else if (is_pix(x, a))
return (pi_[x] - a + 1) * SIGN;

if (max_a_cached_ < min(a, max_a_) &&
(uint64_t) x <= max_x_)
init_cache(min(a, max_a_));

if (is_cached(x, a))
return phi_cache(x, a) * SIGN;

int64_t sum;
int64_t c = PhiTiny::max_a();
int64_t larger_c = min(max_a_cached_, a);
larger_c = std::max(c, larger_c);
ASSERT(c < a);

if (is_cached(x, larger_c))
sum = phi_cache(x, (c = larger_c)) * SIGN;
else
sum = phi_tiny(x, c) * SIGN;

int64_t sqrtx = isqrt(x);
int64_t i;

for (i = c + 1; i <= a; i++)
{
if_unlikely(primes_[i] > sqrtx)
goto phi_1;

int64_t xp = fast_div(x, primes_[i]);

if (is_pix(xp, i - 1))
{
sum += (pi_[xp] - i + 2) * -SIGN;
i += 1; break;
}

if (is_cached(xp, i - 1))
sum += phi_cache(xp, i - 1) * -SIGN;
else
sum += phi<-SIGN>(xp, i - 1);
}

for (; i <= a; i++)
{
if_unlikely(primes_[i] > sqrtx)
goto phi_1;

int64_t xp = fast_div(x, primes_[i]);
ASSERT(is_pix(xp, i - 1));
sum += (pi_[xp] - i + 2) * -SIGN;
}

phi_1:

sum += (a + 1 - i) * -SIGN;
return sum;
}

private:
bool is_pix(uint64_t x, uint64_t a) const
{
return x < pi_.size() &&
x < isquare(primes_[a + 1]);
}

bool is_cached(uint64_t x, uint64_t a) const
{
return x <= max_x_ &&
a <= max_a_cached_ &&
a > PhiTiny::max_a();
}

int64_t phi_cache(uint64_t x, uint64_t a) const
{
ASSERT(is_cached(x, a));
uint64_t count = sieve_[a][x / 240].count;
uint64_t bits = sieve_[a][x / 240].bits;
uint64_t bitmask = unset_larger_[x % 240];
return count + popcnt64(bits & bitmask);
}

void init_cache(uint64_t a)
{
ASSERT(a > PhiTiny::max_a());
ASSERT(a <= max_a_);

if (sieve_.empty())
{
ASSERT(max_a_ >= 3);
sieve_.resize(max_a_ + 1);
sieve_[3].resize(max_x_size_);
std::fill(sieve_[3].begin(), sieve_[3].end(), sieve_t{0, ~0ull});
max_a_cached_ = 3;
}

uint64_t i = max_a_cached_ + 1;
ASSERT(a > max_a_cached_);
max_a_cached_ = a;

for (; i <= a; i++)
{
if (i - 1 <= PhiTiny::max_a())
sieve_[i] = std::move(sieve_[i - 1]);
else
{
sieve_[i].resize(sieve_[i - 1].size());
std::copy(sieve_[i - 1].begin(), sieve_[i - 1].end(), sieve_[i].begin());
}

uint64_t prime = primes_[i];
if (prime <= max_x_)
sieve_[i][prime / 240].bits &= unset_bit_[prime % 240];
for (uint64_t n = prime * prime; n <= max_x_; n += prime * 2)
sieve_[i][n / 240].bits &= unset_bit_[n % 240];

if (i > PhiTiny::max_a())
{
uint64_t count = 0;
for (auto& sieve : sieve_[i])
{
sieve.count = (uint32_t) count;
count += popcnt64(sieve.bits);
}
}
}
}

uint64_t max_x_ = 0;
uint64_t max_x_size_ = 0;
uint64_t max_a_cached_ = 0;
uint64_t max_a_ = 0;

#pragma pack(push, 1)
struct sieve_t
{
uint32_t count;
uint64_t bits;
};
#pragma pack(pop)

pod_vector<pod_vector<sieve_t>> sieve_;
const Primes& primes_;
const PiTable& pi_;
};

template <typename Primes>
pod_vector<int64_t>
generate_phi(int64_t x,
int64_t a,
const Primes& primes,
const PiTable& pi)
{
int64_t size = a + 1;
pod_vector<int64_t> phi(size);
phi[0] = 0;

if (size > 1)
{
if ((int64_t) primes[a] > x)
a = pi[x];

phi[1] = x;
int64_t i = 2;
int64_t sqrtx = isqrt(x);
PhiCache<Primes> cache(x, a, primes, pi);

for (; i <= a && primes[i - 1] <= sqrtx; i++)
phi[i] = phi[i - 1] + cache.template phi<-1>(x / primes[i - 1], i - 2);

for (; i <= a; i++)
phi[i] = phi[i - 1] - (x > 0);

for (; i < size; i++)
phi[i] = x > 0;
}

return phi;
}

} 

#endif
