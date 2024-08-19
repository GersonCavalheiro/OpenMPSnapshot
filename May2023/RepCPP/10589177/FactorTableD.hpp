
#ifndef FACTORTABLED_HPP
#define FACTORTABLED_HPP

#include <primecount.hpp>
#include <primecount-internal.hpp>
#include <BaseFactorTable.hpp>
#include <primesieve.hpp>
#include <imath.hpp>
#include <int128_t.hpp>
#include <macros.hpp>
#include <pod_vector.hpp>

#include <algorithm>
#include <limits>
#include <stdint.h>

namespace {

using namespace primecount;

template <typename T>
class FactorTableD : public BaseFactorTable
{
public:
FactorTableD(int64_t y,
int64_t z,
int threads)
{
if_unlikely(z > max())
throw primecount_error("z must be <= FactorTable::max()");

z = std::max<int64_t>(1, z);
T T_MAX = std::numeric_limits<T>::max();
factor_.resize(to_index(z) + 1);

factor_[0] = T_MAX ^ 1;

int64_t sqrtz = isqrt(z);
int64_t thread_threshold = (int64_t) 1e7;
threads = ideal_num_threads(z, threads, thread_threshold);
int64_t thread_distance = ceil_div(z, threads);
thread_distance += coprime_indexes_.size() - thread_distance % coprime_indexes_.size();

#pragma omp parallel for num_threads(threads)
for (int t = 0; t < threads; t++)
{
int64_t low = thread_distance * t;
int64_t high = low + thread_distance;
low = std::max(first_coprime(), low + 1);
high = std::min(high, z);

if (low <= high)
{
int64_t low_idx = to_index(low);
int64_t size = (to_index(high) + 1) - low_idx;
std::fill_n(&factor_[low_idx], size, T_MAX);

int64_t start = first_coprime();
int64_t stop = high / first_coprime();
int64_t min_m = first_coprime() * first_coprime();
primesieve::iterator it(start, stop);

if (min_m <= high)
{
while (true)
{
int64_t i = 1;
int64_t prime = it.next_prime();
int64_t multiple = next_multiple(prime, low, &i);
min_m = prime * first_coprime();

if (min_m > high)
break;

for (; multiple <= high; multiple = prime * to_number(i++))
{
int64_t mi = to_index(multiple);
if (factor_[mi] == T_MAX)
factor_[mi] = (T) prime;
else if (factor_[mi] != 0)
factor_[mi] ^= 1;
}

if (prime <= sqrtz)
{
int64_t j = 0;
int64_t square = prime * prime;
multiple = next_multiple(square, low, &j);

for (; multiple <= high; multiple = square * to_number(j++))
factor_[to_index(multiple)] = 0;
}
}
}

start = std::max(start, y + 1);

if (start <= high)
{
it.jump_to(start, high);

while (true)
{
int64_t i = 0;
int64_t prime = it.next_prime();
int64_t next = next_multiple(prime, low, &i);

if (prime > high)
break;

for (; next <= high; next = prime * to_number(i++))
factor_[to_index(next)] = 0;
}
}
}
}
}

int64_t is_leaf(int64_t index) const
{
return factor_[index];
}

int64_t mu(int64_t index) const
{
#if defined(ENABLE_MU_0_TESTING)
if (factor_[index] == 0)
return 0;
#else
ASSERT(factor_[index] != 0);
#endif

if (factor_[index] & 1)
return -1;
else
return 1;
}

static maxint_t max()
{
maxint_t T_MAX = std::numeric_limits<T>::max();
return ipow(T_MAX - 1, 2) - 1;
}

private:
pod_vector<T> factor_;
};

} 

#endif
