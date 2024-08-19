
#ifndef PHITINY_HPP
#define PHITINY_HPP

#include <BitSieve240.hpp>
#include <fast_div.hpp>
#include <imath.hpp>
#include <macros.hpp>
#include <pod_vector.hpp>
#include <popcnt.hpp>

#include <stdint.h>
#include <limits>
#include <type_traits>

namespace primecount {

class PhiTiny : public BitSieve240
{
public:
PhiTiny();

template <typename T>
T phi_recursive(T x, uint64_t a) const
{
using UT = typename std::make_unsigned<T>::type;

if (a < max_a())
return phi((UT) x, a);
else
{
ASSERT(a == 8);
return phi7((UT) x) - phi7((UT) x / 19);
}
}

template <typename T>
T phi(T x, uint64_t a) const
{
auto pp = prime_products[a];
auto remainder = (uint64_t)(x % pp);
T xpp = x / pp;
T sum = xpp * totients[a];

if (a < phi_.size())
sum += phi_[a][remainder];
else
{
uint64_t count = sieve_[a][remainder / 240].count;
uint64_t bits = sieve_[a][remainder / 240].bits;
uint64_t bitmask = unset_larger_[remainder % 240];
sum += (T)(count + popcnt64(bits & bitmask));
}

return sum;
}

template <typename T>
T phi7(T x) const
{
constexpr uint32_t a = 7;
constexpr uint32_t pp = 510510;
constexpr uint32_t totient = 92160;
auto remainder = (uint64_t)(x % pp);
T xpp = x / pp;
T sum = xpp * totient;

ASSERT(sieve_.size() - 1 == a);
uint64_t count = sieve_[a][remainder / 240].count;
uint64_t bits = sieve_[a][remainder / 240].bits;
uint64_t bitmask = unset_larger_[remainder % 240];
sum += (T)(count + popcnt64(bits & bitmask));

return sum;
}

static uint64_t get_c(uint64_t y)
{
if (y < pi.size())
return pi[y];
else
return max_a();
}

template <typename T>
static uint64_t get_k(T x)
{
return get_c(iroot<4>(x));
}

static constexpr uint64_t max_a()
{
return primes.size();
}

private:
static const pod_array<uint32_t, 8> primes;
static const pod_array<uint32_t, 8> prime_products;
static const pod_array<uint32_t, 8> totients;
static const pod_array<uint8_t, 20> pi;

#pragma pack(push, 1)
struct sieve_t
{
uint32_t count;
uint64_t bits;
};

#pragma pack(pop)

pod_array<pod_vector<sieve_t>, 8> sieve_;
pod_array<pod_vector<uint8_t>, 4> phi_;
};

extern const PhiTiny phiTiny;

inline bool is_phi_tiny(uint64_t a)
{
return a <= PhiTiny::max_a();
}

template <typename T>
typename std::enable_if<(sizeof(T) == sizeof(typename make_smaller<T>::type)), T>::type
phi_tiny(T x, uint64_t a)
{
return phiTiny.phi_recursive(x, a);
}

template <typename T>
typename std::enable_if<(sizeof(T) > sizeof(typename make_smaller<T>::type)), T>::type
phi_tiny(T x, uint64_t a)
{
using smaller_t = typename make_smaller<T>::type;

if (x <= std::numeric_limits<smaller_t>::max())
return phiTiny.phi_recursive((smaller_t) x, a);
else
return phiTiny.phi_recursive(x, a);
}

} 

#endif
