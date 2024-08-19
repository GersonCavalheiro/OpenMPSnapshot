#pragma once

#include <array> 
#include <cassert> 
#include <cmath>   
#include <cstdint> 
#include <cstring> 
#include <limits> 
#include <type_traits> 

#include <nlohmann/detail/boolean_operators.hpp>
#include <nlohmann/detail/macro_scope.hpp>

namespace nlohmann
{
namespace detail
{


namespace dtoa_impl
{

template <typename Target, typename Source>
Target reinterpret_bits(const Source source)
{
static_assert(sizeof(Target) == sizeof(Source), "size mismatch");

Target target;
std::memcpy(&target, &source, sizeof(Source));
return target;
}

struct diyfp 
{
static constexpr int kPrecision = 64; 

std::uint64_t f = 0;
int e = 0;

constexpr diyfp(std::uint64_t f_, int e_) noexcept : f(f_), e(e_) {}


static diyfp sub(const diyfp& x, const diyfp& y) noexcept
{
assert(x.e == y.e);
assert(x.f >= y.f);

return {x.f - y.f, x.e};
}


static diyfp mul(const diyfp& x, const diyfp& y) noexcept
{
static_assert(kPrecision == 64, "internal error");



const std::uint64_t u_lo = x.f & 0xFFFFFFFFu;
const std::uint64_t u_hi = x.f >> 32u;
const std::uint64_t v_lo = y.f & 0xFFFFFFFFu;
const std::uint64_t v_hi = y.f >> 32u;

const std::uint64_t p0 = u_lo * v_lo;
const std::uint64_t p1 = u_lo * v_hi;
const std::uint64_t p2 = u_hi * v_lo;
const std::uint64_t p3 = u_hi * v_hi;

const std::uint64_t p0_hi = p0 >> 32u;
const std::uint64_t p1_lo = p1 & 0xFFFFFFFFu;
const std::uint64_t p1_hi = p1 >> 32u;
const std::uint64_t p2_lo = p2 & 0xFFFFFFFFu;
const std::uint64_t p2_hi = p2 >> 32u;

std::uint64_t Q = p0_hi + p1_lo + p2_lo;


Q += std::uint64_t{1} << (64u - 32u - 1u); 

const std::uint64_t h = p3 + p2_hi + p1_hi + (Q >> 32u);

return {h, x.e + y.e + 64};
}


static diyfp normalize(diyfp x) noexcept
{
assert(x.f != 0);

while ((x.f >> 63u) == 0)
{
x.f <<= 1u;
x.e--;
}

return x;
}


static diyfp normalize_to(const diyfp& x, const int target_exponent) noexcept
{
const int delta = x.e - target_exponent;

assert(delta >= 0);
assert(((x.f << delta) >> delta) == x.f);

return {x.f << delta, target_exponent};
}
};

struct boundaries
{
diyfp w;
diyfp minus;
diyfp plus;
};


template <typename FloatType>
boundaries compute_boundaries(FloatType value)
{
assert(std::isfinite(value));
assert(value > 0);


static_assert(std::numeric_limits<FloatType>::is_iec559,
"internal error: dtoa_short requires an IEEE-754 floating-point implementation");

constexpr int      kPrecision = std::numeric_limits<FloatType>::digits; 
constexpr int      kBias      = std::numeric_limits<FloatType>::max_exponent - 1 + (kPrecision - 1);
constexpr int      kMinExp    = 1 - kBias;
constexpr std::uint64_t kHiddenBit = std::uint64_t{1} << (kPrecision - 1); 

using bits_type = typename std::conditional<kPrecision == 24, std::uint32_t, std::uint64_t >::type;

const std::uint64_t bits = reinterpret_bits<bits_type>(value);
const std::uint64_t E = bits >> (kPrecision - 1);
const std::uint64_t F = bits & (kHiddenBit - 1);

const bool is_denormal = E == 0;
const diyfp v = is_denormal
? diyfp(F, kMinExp)
: diyfp(F + kHiddenBit, static_cast<int>(E) - kBias);


const bool lower_boundary_is_closer = F == 0 and E > 1;
const diyfp m_plus = diyfp(2 * v.f + 1, v.e - 1);
const diyfp m_minus = lower_boundary_is_closer
? diyfp(4 * v.f - 1, v.e - 2)  
: diyfp(2 * v.f - 1, v.e - 1); 

const diyfp w_plus = diyfp::normalize(m_plus);

const diyfp w_minus = diyfp::normalize_to(m_minus, w_plus.e);

return {diyfp::normalize(v), w_minus, w_plus};
}


constexpr int kAlpha = -60;
constexpr int kGamma = -32;

struct cached_power 
{
std::uint64_t f;
int e;
int k;
};


inline cached_power get_cached_power_for_binary_exponent(int e)
{


constexpr int kCachedPowersMinDecExp = -300;
constexpr int kCachedPowersDecStep = 8;

static constexpr std::array<cached_power, 79> kCachedPowers =
{
{
{ 0xAB70FE17C79AC6CA, -1060, -300 },
{ 0xFF77B1FCBEBCDC4F, -1034, -292 },
{ 0xBE5691EF416BD60C, -1007, -284 },
{ 0x8DD01FAD907FFC3C,  -980, -276 },
{ 0xD3515C2831559A83,  -954, -268 },
{ 0x9D71AC8FADA6C9B5,  -927, -260 },
{ 0xEA9C227723EE8BCB,  -901, -252 },
{ 0xAECC49914078536D,  -874, -244 },
{ 0x823C12795DB6CE57,  -847, -236 },
{ 0xC21094364DFB5637,  -821, -228 },
{ 0x9096EA6F3848984F,  -794, -220 },
{ 0xD77485CB25823AC7,  -768, -212 },
{ 0xA086CFCD97BF97F4,  -741, -204 },
{ 0xEF340A98172AACE5,  -715, -196 },
{ 0xB23867FB2A35B28E,  -688, -188 },
{ 0x84C8D4DFD2C63F3B,  -661, -180 },
{ 0xC5DD44271AD3CDBA,  -635, -172 },
{ 0x936B9FCEBB25C996,  -608, -164 },
{ 0xDBAC6C247D62A584,  -582, -156 },
{ 0xA3AB66580D5FDAF6,  -555, -148 },
{ 0xF3E2F893DEC3F126,  -529, -140 },
{ 0xB5B5ADA8AAFF80B8,  -502, -132 },
{ 0x87625F056C7C4A8B,  -475, -124 },
{ 0xC9BCFF6034C13053,  -449, -116 },
{ 0x964E858C91BA2655,  -422, -108 },
{ 0xDFF9772470297EBD,  -396, -100 },
{ 0xA6DFBD9FB8E5B88F,  -369,  -92 },
{ 0xF8A95FCF88747D94,  -343,  -84 },
{ 0xB94470938FA89BCF,  -316,  -76 },
{ 0x8A08F0F8BF0F156B,  -289,  -68 },
{ 0xCDB02555653131B6,  -263,  -60 },
{ 0x993FE2C6D07B7FAC,  -236,  -52 },
{ 0xE45C10C42A2B3B06,  -210,  -44 },
{ 0xAA242499697392D3,  -183,  -36 },
{ 0xFD87B5F28300CA0E,  -157,  -28 },
{ 0xBCE5086492111AEB,  -130,  -20 },
{ 0x8CBCCC096F5088CC,  -103,  -12 },
{ 0xD1B71758E219652C,   -77,   -4 },
{ 0x9C40000000000000,   -50,    4 },
{ 0xE8D4A51000000000,   -24,   12 },
{ 0xAD78EBC5AC620000,     3,   20 },
{ 0x813F3978F8940984,    30,   28 },
{ 0xC097CE7BC90715B3,    56,   36 },
{ 0x8F7E32CE7BEA5C70,    83,   44 },
{ 0xD5D238A4ABE98068,   109,   52 },
{ 0x9F4F2726179A2245,   136,   60 },
{ 0xED63A231D4C4FB27,   162,   68 },
{ 0xB0DE65388CC8ADA8,   189,   76 },
{ 0x83C7088E1AAB65DB,   216,   84 },
{ 0xC45D1DF942711D9A,   242,   92 },
{ 0x924D692CA61BE758,   269,  100 },
{ 0xDA01EE641A708DEA,   295,  108 },
{ 0xA26DA3999AEF774A,   322,  116 },
{ 0xF209787BB47D6B85,   348,  124 },
{ 0xB454E4A179DD1877,   375,  132 },
{ 0x865B86925B9BC5C2,   402,  140 },
{ 0xC83553C5C8965D3D,   428,  148 },
{ 0x952AB45CFA97A0B3,   455,  156 },
{ 0xDE469FBD99A05FE3,   481,  164 },
{ 0xA59BC234DB398C25,   508,  172 },
{ 0xF6C69A72A3989F5C,   534,  180 },
{ 0xB7DCBF5354E9BECE,   561,  188 },
{ 0x88FCF317F22241E2,   588,  196 },
{ 0xCC20CE9BD35C78A5,   614,  204 },
{ 0x98165AF37B2153DF,   641,  212 },
{ 0xE2A0B5DC971F303A,   667,  220 },
{ 0xA8D9D1535CE3B396,   694,  228 },
{ 0xFB9B7CD9A4A7443C,   720,  236 },
{ 0xBB764C4CA7A44410,   747,  244 },
{ 0x8BAB8EEFB6409C1A,   774,  252 },
{ 0xD01FEF10A657842C,   800,  260 },
{ 0x9B10A4E5E9913129,   827,  268 },
{ 0xE7109BFBA19C0C9D,   853,  276 },
{ 0xAC2820D9623BF429,   880,  284 },
{ 0x80444B5E7AA7CF85,   907,  292 },
{ 0xBF21E44003ACDD2D,   933,  300 },
{ 0x8E679C2F5E44FF8F,   960,  308 },
{ 0xD433179D9C8CB841,   986,  316 },
{ 0x9E19DB92B4E31BA9,  1013,  324 },
}
};

assert(e >= -1500);
assert(e <=  1500);
const int f = kAlpha - e - 1;
const int k = (f * 78913) / (1 << 18) + static_cast<int>(f > 0);

const int index = (-kCachedPowersMinDecExp + k + (kCachedPowersDecStep - 1)) / kCachedPowersDecStep;
assert(index >= 0);
assert(static_cast<std::size_t>(index) < kCachedPowers.size());

const cached_power cached = kCachedPowers[static_cast<std::size_t>(index)];
assert(kAlpha <= cached.e + e + 64);
assert(kGamma >= cached.e + e + 64);

return cached;
}


inline int find_largest_pow10(const std::uint32_t n, std::uint32_t& pow10)
{
if (n >= 1000000000)
{
pow10 = 1000000000;
return 10;
}
else if (n >= 100000000)
{
pow10 = 100000000;
return  9;
}
else if (n >= 10000000)
{
pow10 = 10000000;
return  8;
}
else if (n >= 1000000)
{
pow10 = 1000000;
return  7;
}
else if (n >= 100000)
{
pow10 = 100000;
return  6;
}
else if (n >= 10000)
{
pow10 = 10000;
return  5;
}
else if (n >= 1000)
{
pow10 = 1000;
return  4;
}
else if (n >= 100)
{
pow10 = 100;
return  3;
}
else if (n >= 10)
{
pow10 = 10;
return  2;
}
else
{
pow10 = 1;
return 1;
}
}

inline void grisu2_round(char* buf, int len, std::uint64_t dist, std::uint64_t delta,
std::uint64_t rest, std::uint64_t ten_k)
{
assert(len >= 1);
assert(dist <= delta);
assert(rest <= delta);
assert(ten_k > 0);



while (rest < dist
and delta - rest >= ten_k
and (rest + ten_k < dist or dist - rest > rest + ten_k - dist))
{
assert(buf[len - 1] != '0');
buf[len - 1]--;
rest += ten_k;
}
}


inline void grisu2_digit_gen(char* buffer, int& length, int& decimal_exponent,
diyfp M_minus, diyfp w, diyfp M_plus)
{
static_assert(kAlpha >= -60, "internal error");
static_assert(kGamma <= -32, "internal error");


assert(M_plus.e >= kAlpha);
assert(M_plus.e <= kGamma);

std::uint64_t delta = diyfp::sub(M_plus, M_minus).f; 
std::uint64_t dist  = diyfp::sub(M_plus, w      ).f; 


const diyfp one(std::uint64_t{1} << -M_plus.e, M_plus.e);

auto p1 = static_cast<std::uint32_t>(M_plus.f >> -one.e); 
std::uint64_t p2 = M_plus.f & (one.f - 1);                    


assert(p1 > 0);

std::uint32_t pow10;
const int k = find_largest_pow10(p1, pow10);


int n = k;
while (n > 0)
{
const std::uint32_t d = p1 / pow10;  
const std::uint32_t r = p1 % pow10;  
assert(d <= 9);
buffer[length++] = static_cast<char>('0' + d); 
p1 = r;
n--;

const std::uint64_t rest = (std::uint64_t{p1} << -one.e) + p2;
if (rest <= delta)
{

decimal_exponent += n;

const std::uint64_t ten_n = std::uint64_t{pow10} << -one.e;
grisu2_round(buffer, length, dist, delta, rest, ten_n);

return;
}

pow10 /= 10;
}


assert(p2 > delta);

int m = 0;
for (;;)
{
assert(p2 <= (std::numeric_limits<std::uint64_t>::max)() / 10);
p2 *= 10;
const std::uint64_t d = p2 >> -one.e;     
const std::uint64_t r = p2 & (one.f - 1); 
assert(d <= 9);
buffer[length++] = static_cast<char>('0' + d); 
p2 = r;
m++;

delta *= 10;
dist  *= 10;
if (p2 <= delta)
{
break;
}
}


decimal_exponent -= m;

const std::uint64_t ten_m = one.f;
grisu2_round(buffer, length, dist, delta, p2, ten_m);

}


JSON_HEDLEY_NON_NULL(1)
inline void grisu2(char* buf, int& len, int& decimal_exponent,
diyfp m_minus, diyfp v, diyfp m_plus)
{
assert(m_plus.e == m_minus.e);
assert(m_plus.e == v.e);


const cached_power cached = get_cached_power_for_binary_exponent(m_plus.e);

const diyfp c_minus_k(cached.f, cached.e); 

const diyfp w       = diyfp::mul(v,       c_minus_k);
const diyfp w_minus = diyfp::mul(m_minus, c_minus_k);
const diyfp w_plus  = diyfp::mul(m_plus,  c_minus_k);

const diyfp M_minus(w_minus.f + 1, w_minus.e);
const diyfp M_plus (w_plus.f  - 1, w_plus.e );

decimal_exponent = -cached.k; 

grisu2_digit_gen(buf, len, decimal_exponent, M_minus, w, M_plus);
}


template <typename FloatType>
JSON_HEDLEY_NON_NULL(1)
void grisu2(char* buf, int& len, int& decimal_exponent, FloatType value)
{
static_assert(diyfp::kPrecision >= std::numeric_limits<FloatType>::digits + 3,
"internal error: not enough precision");

assert(std::isfinite(value));
assert(value > 0);

#if 0
const boundaries w = compute_boundaries(static_cast<double>(value));
#else
const boundaries w = compute_boundaries(value);
#endif

grisu2(buf, len, decimal_exponent, w.minus, w.w, w.plus);
}


JSON_HEDLEY_NON_NULL(1)
JSON_HEDLEY_RETURNS_NON_NULL
inline char* append_exponent(char* buf, int e)
{
assert(e > -1000);
assert(e <  1000);

if (e < 0)
{
e = -e;
*buf++ = '-';
}
else
{
*buf++ = '+';
}

auto k = static_cast<std::uint32_t>(e);
if (k < 10)
{
*buf++ = '0';
*buf++ = static_cast<char>('0' + k);
}
else if (k < 100)
{
*buf++ = static_cast<char>('0' + k / 10);
k %= 10;
*buf++ = static_cast<char>('0' + k);
}
else
{
*buf++ = static_cast<char>('0' + k / 100);
k %= 100;
*buf++ = static_cast<char>('0' + k / 10);
k %= 10;
*buf++ = static_cast<char>('0' + k);
}

return buf;
}


JSON_HEDLEY_NON_NULL(1)
JSON_HEDLEY_RETURNS_NON_NULL
inline char* format_buffer(char* buf, int len, int decimal_exponent,
int min_exp, int max_exp)
{
assert(min_exp < 0);
assert(max_exp > 0);

const int k = len;
const int n = len + decimal_exponent;


if (k <= n and n <= max_exp)
{

std::memset(buf + k, '0', static_cast<size_t>(n) - static_cast<size_t>(k));
buf[n + 0] = '.';
buf[n + 1] = '0';
return buf + (static_cast<size_t>(n) + 2);
}

if (0 < n and n <= max_exp)
{

assert(k > n);

std::memmove(buf + (static_cast<size_t>(n) + 1), buf + n, static_cast<size_t>(k) - static_cast<size_t>(n));
buf[n] = '.';
return buf + (static_cast<size_t>(k) + 1U);
}

if (min_exp < n and n <= 0)
{

std::memmove(buf + (2 + static_cast<size_t>(-n)), buf, static_cast<size_t>(k));
buf[0] = '0';
buf[1] = '.';
std::memset(buf + 2, '0', static_cast<size_t>(-n));
return buf + (2U + static_cast<size_t>(-n) + static_cast<size_t>(k));
}

if (k == 1)
{

buf += 1;
}
else
{

std::memmove(buf + 2, buf + 1, static_cast<size_t>(k) - 1);
buf[1] = '.';
buf += 1 + static_cast<size_t>(k);
}

*buf++ = 'e';
return append_exponent(buf, n - 1);
}

} 


template <typename FloatType>
JSON_HEDLEY_NON_NULL(1, 2)
JSON_HEDLEY_RETURNS_NON_NULL
char* to_chars(char* first, const char* last, FloatType value)
{
static_cast<void>(last); 
assert(std::isfinite(value));

if (std::signbit(value))
{
value = -value;
*first++ = '-';
}

if (value == 0) 
{
*first++ = '0';
*first++ = '.';
*first++ = '0';
return first;
}

assert(last - first >= std::numeric_limits<FloatType>::max_digits10);

int len = 0;
int decimal_exponent = 0;
dtoa_impl::grisu2(first, len, decimal_exponent, value);

assert(len <= std::numeric_limits<FloatType>::max_digits10);

constexpr int kMinExp = -4;
constexpr int kMaxExp = std::numeric_limits<FloatType>::digits10;

assert(last - first >= kMaxExp + 2);
assert(last - first >= 2 + (-kMinExp - 1) + std::numeric_limits<FloatType>::max_digits10);
assert(last - first >= std::numeric_limits<FloatType>::max_digits10 + 6);

return dtoa_impl::format_buffer(first, len, decimal_exponent, kMinExp, kMaxExp);
}

} 
} 
