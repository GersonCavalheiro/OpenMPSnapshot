
#ifndef STOREPRIMES_HPP
#define STOREPRIMES_HPP

#include "iterator.hpp"
#include "primesieve_error.hpp"

#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#if defined(min) || defined(max)
#undef min
#undef max
#if __cplusplus >= 202301L
#warning "Undefining min()/max() macros. Please define NOMINMAX before including <Windows.h>"
#elif defined(_MSC_VER) || defined(__GNUG__)
#pragma message("Undefining min()/max() macros. Please define NOMINMAX before including <Windows.h>")
#endif
#endif

namespace primesieve {

inline std::size_t prime_count_approx(uint64_t start, uint64_t stop)
{
if (start > stop)
return 0;

double x = (double) stop;
x = std::max(100.0, x);
double pix = (stop - start) / (std::log(x) - 1.1) + 5;

return (std::size_t) pix;
}

template <typename T> inline std::string getTypeName() { return "Type"; }
template <> inline std::string getTypeName<int8_t>() { return "int8_t"; }
template <> inline std::string getTypeName<uint8_t>() { return "uint8_t"; }
template <> inline std::string getTypeName<int16_t>() { return "int16_t"; }
template <> inline std::string getTypeName<uint16_t>() { return "uint16_t"; }
template <> inline std::string getTypeName<int32_t>() { return "int32_t"; }
template <> inline std::string getTypeName<uint32_t>() { return "uint32_t"; }
template <> inline std::string getTypeName<int64_t>() { return "int64_t"; }
template <> inline std::string getTypeName<uint64_t>() { return "uint64_t"; }

template <typename T>
inline void store_primes(uint64_t start,
uint64_t stop,
T& primes)
{
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4018)
#endif

if (start > stop)
return;

uint64_t maxPrime64bits = 18446744073709551557ull;
if (start > maxPrime64bits)
return;

using V = typename T::value_type;
if (stop > std::numeric_limits<V>::max())
throw primesieve_error("store_primes(): " + getTypeName<V>() + " is too narrow for generating primes up to " + std::to_string(stop));

std::size_t size = primes.size() + prime_count_approx(start, stop);
primes.reserve(size);

primesieve::iterator it(start, stop);
it.generate_next_primes();

uint64_t limit = std::min(stop, maxPrime64bits - 1);

for (; it.primes_[it.size_ - 1] <= limit; it.generate_next_primes())
primes.insert(primes.end(), it.primes_, it.primes_ + it.size_);
for (std::size_t i = 0; it.primes_[i] <= limit; i++)
primes.push_back((V) it.primes_[i]);

if (stop >= maxPrime64bits)
primes.push_back((V) maxPrime64bits);

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

template <typename T>
inline void store_n_primes(uint64_t n,
uint64_t start,
T& primes)
{
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4018)
#endif

if (n == 0)
return;

using V = typename T::value_type;
std::size_t size = primes.size() + (std::size_t) n;
primes.reserve(size);

double x = std::max({6.0, (double) n, (double) start});
double logn = std::log(x);
double loglogn = std::log(logn);
uint64_t nthPrime = (uint64_t)(n * (logn + loglogn));
uint64_t stop = start + nthPrime;

primesieve::iterator it(start, stop);
it.generate_next_primes();

while (n >= it.size_)
{
if (it.primes_[it.size_ - 1] > std::numeric_limits<V>::max())
throw primesieve_error("store_n_primes(): " + getTypeName<V>() + " is too narrow for generating primes up to " + std::to_string(stop));

primes.insert(primes.end(), it.primes_, it.primes_ + it.size_);
n -= it.size_;
if (n == 0)
return;

it.generate_next_primes();
}

if (it.primes_[n - 1] > std::numeric_limits<V>::max())
throw primesieve_error("store_n_primes(): " + getTypeName<V>() + " is too narrow for generating primes up to " + std::to_string(stop));

for (std::size_t i = 0; i < (std::size_t) n; i++)
primes.push_back((V) it.primes_[i]);

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

} 

#endif
