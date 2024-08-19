
#ifndef FACTORTABLE_HPP
#define FACTORTABLE_HPP

#include <primesum.hpp>
#include <primesum-internal.hpp>
#include <primesieve.hpp>
#include <imath.hpp>
#include <int128_t.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdint.h>
#include <vector>

namespace primesum {

class AbstractFactorTable
{
protected:
virtual ~AbstractFactorTable() { }

public:
static void to_index(int64_t* number)
{
assert(*number > 0);
*number = get_index(*number);
}

static int64_t get_index(uint64_t number)
{
assert(number > 0);
uint64_t q = number / 210;
uint64_t r = number % 210;
return 48 * q + indexes_[r];
}

static int64_t get_number(uint64_t index)
{
uint64_t q = index / 48;
uint64_t r = index % 48;
return 210 * q + numbers_[r];
}

private:
static const uint8_t numbers_[48];
static const  int8_t indexes_[210];
};

template <typename T>
class FactorTable : public AbstractFactorTable
{
public:
FactorTable(int64_t y, int threads)
{
if (y > max())
throw primesum_error("y must be <= FactorTable::max()");

y = std::max<int64_t>(8, y);
T T_MAX = std::numeric_limits<T>::max();
factor_.resize(get_index(y) + 1, T_MAX);

int64_t sqrty = isqrt(y);
int64_t thread_threshold = ipow(10, 7);
threads = ideal_num_threads(threads, y, thread_threshold);
int64_t thread_distance = ceil_div(y, threads);

#pragma omp parallel for num_threads(threads)
for (int t = 0; t < threads; t++)
{
int64_t low = 1;
low += thread_distance * t;
int64_t high = std::min(low + thread_distance, y);
primesieve::iterator it(get_number(1) - 1);

while (true)
{
int64_t i = 1;
int64_t prime = it.next_prime();
int64_t multiple = next_multiple(prime, low, &i);
int64_t min_m = prime * get_number(1);

if (min_m > high)
break;

for (; multiple <= high; multiple = prime * get_number(i++))
{
int64_t mi = get_index(multiple);
if (factor_[mi] == T_MAX)
factor_[mi] = (T) prime;
else if (factor_[mi] != 0)
factor_[mi] ^= 1;
}

if (prime <= sqrty)
{
int64_t j = 0;
int64_t square = prime * prime;
multiple = next_multiple(square, low, &j);

for (; multiple <= high; multiple = square * get_number(j++))
factor_[get_index(multiple)] = 0;
}
}
}
}

int64_t lpf(int64_t index) const
{
return factor_[index];
}

int64_t mu(int64_t index) const
{
assert(factor_[index] != 0);
return (factor_[index] & 1) ? -1 : 1;
}

static int128_t max()
{
int128_t T_MAX = std::numeric_limits<T>::max();
return ipow(T_MAX - 1, 2) - 1;
}

private:

static int64_t next_multiple(int64_t prime,
int64_t low,
int64_t* index)
{
int64_t quotient = ceil_div(low, prime);
int64_t i = std::max(*index, get_index(quotient));
int64_t multiple = 0;

for (; multiple <= low; i++)
multiple = prime * get_number(i);

*index = i;
return multiple;
}

std::vector<T> factor_;
};

} 

#endif
