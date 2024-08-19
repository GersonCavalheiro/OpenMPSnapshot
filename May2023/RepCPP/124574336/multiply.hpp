#ifndef BOOST_MP_CPP_INT_MUL_HPP
#define BOOST_MP_CPP_INT_MUL_HPP

#include <boost/multiprecision/integer.hpp>

namespace boost { namespace multiprecision { namespace backends {

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) 
#endif
template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2>
inline BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const limb_type&                                                            val) BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value))
{
if (!val)
{
result = static_cast<limb_type>(0);
return;
}
if ((void*)&a != (void*)&result)
result.resize(a.size(), a.size());
double_limb_type                                                                                  carry = 0;
typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::limb_pointer       p     = result.limbs();
typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::limb_pointer       pe    = result.limbs() + result.size();
typename cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>::const_limb_pointer pa    = a.limbs();
while (p != pe)
{
carry += static_cast<double_limb_type>(*pa) * static_cast<double_limb_type>(val);
#ifdef __MSVC_RUNTIME_CHECKS
*p = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
#else
*p = static_cast<limb_type>(carry);
#endif
carry >>= cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::limb_bits;
++p, ++pa;
}
if (carry)
{
unsigned i = result.size();
result.resize(i + 1, i + 1);
if (result.size() > i)
result.limbs()[i] = static_cast<limb_type>(carry);
}
result.sign(a.sign());
if (is_fixed_precision<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value)
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
inline BOOST_MP_CXX14_CONSTEXPR void resize_for_carry(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& , unsigned ) {}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, class Allocator1>
inline BOOST_MP_CXX14_CONSTEXPR void resize_for_carry(cpp_int_backend<MinBits1, MaxBits1, SignType1, checked, Allocator1>& result, unsigned required)
{
if (result.size() < required)
result.resize(required, required);
}
#ifdef BOOST_MP_KARATSUBA_CUTOFF
const size_t karatsuba_cutoff = BOOST_MP_KARATSUBA_CUTOFF;
#else
const size_t karatsuba_cutoff = 40;
#endif
template <unsigned MinBits, unsigned MaxBits, cpp_int_check_type Checked, class Allocator>
inline void multiply_karatsuba(
cpp_int_backend<MinBits, MaxBits, signed_magnitude, Checked, Allocator>&       result,
const cpp_int_backend<MinBits, MaxBits, signed_magnitude, Checked, Allocator>& a,
const cpp_int_backend<MinBits, MaxBits, signed_magnitude, Checked, Allocator>& b,
typename cpp_int_backend<MinBits, MaxBits, signed_magnitude, Checked, Allocator>::scoped_shared_storage& storage)
{
typedef cpp_int_backend<MinBits, MaxBits, signed_magnitude, Checked, Allocator> cpp_int_type;

unsigned as = a.size();
unsigned bs = b.size();
if ((as < karatsuba_cutoff) || (bs < karatsuba_cutoff))
{
eval_multiply(result, a, b);
return;
}
unsigned n  = (as > bs ? as : bs) / 2 + 1;
unsigned          sz = (std::min)(as, n);
const cpp_int_type a_l(a.limbs(), 0, sz);

sz = (std::min)(bs, n);
const cpp_int_type b_l(b.limbs(), 0, sz);

limb_type          zero = 0;
const cpp_int_type a_h(as > n ? a.limbs() + n : &zero, 0, as > n ? as - n : 1);
const cpp_int_type b_h(bs > n ? b.limbs() + n : &zero, 0, bs > n ? bs - n : 1);
cpp_int_type t1(storage, 2 * n + 2);
cpp_int_type t2(storage, n + 1);
cpp_int_type t3(storage, n + 1);
cpp_int_type result_low(result.limbs(), 0, 2 * n);
cpp_int_type result_high(result.limbs(), 2 * n, result.size() - 2 * n);
multiply_karatsuba(result_low, a_l, b_l, storage);
for (unsigned i = result_low.size(); i < 2 * n; ++i)
result.limbs()[i] = 0;
multiply_karatsuba(result_high, a_h, b_h, storage);
for (unsigned i = result_high.size() + 2 * n; i < result.size(); ++i)
result.limbs()[i] = 0;
add_unsigned(t2, a_l, a_h);
add_unsigned(t3, b_l, b_h);
multiply_karatsuba(t1, t2, t3, storage); 
subtract_unsigned(t1, t1, result_high);
subtract_unsigned(t1, t1, result_low);
cpp_int_type result_alias(result.limbs(), n, result.size() - n);
add_unsigned(result_alias, result_alias, t1);
storage.deallocate(t1.capacity() + t2.capacity() + t3.capacity());

result.normalize();
}

inline unsigned karatsuba_storage_size(unsigned s)
{
return 5 * s;
}
template <unsigned MinBits, unsigned MaxBits, cpp_integer_type SignType, cpp_int_check_type Checked, class Allocator>
inline typename enable_if_c<!is_fixed_precision<cpp_int_backend<MinBits, MaxBits, SignType, Checked, Allocator> >::value>::type
setup_karatsuba(
cpp_int_backend<MinBits, MaxBits, SignType, Checked, Allocator>& result,
const cpp_int_backend<MinBits, MaxBits, SignType, Checked, Allocator>& a,
const cpp_int_backend<MinBits, MaxBits, SignType, Checked, Allocator>& b)
{
unsigned as = a.size();
unsigned bs = b.size();
unsigned s = as > bs ? as : bs;
unsigned storage_size = karatsuba_storage_size(s);
if (storage_size < 300)
{
limb_type limbs[300];
typename cpp_int_backend<MinBits, MaxBits, SignType, Checked, Allocator>::scoped_shared_storage storage(limbs, storage_size);
multiply_karatsuba(result, a, b, storage);
}
else
{
typename cpp_int_backend<MinBits, MaxBits, SignType, Checked, Allocator>::scoped_shared_storage storage(result.allocator(), storage_size);
multiply_karatsuba(result, a, b, storage);
}
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2, unsigned MinBits3, unsigned MaxBits3, cpp_integer_type SignType3, cpp_int_check_type Checked3, class Allocator3>
inline typename enable_if_c<is_fixed_precision<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value || is_fixed_precision<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value || is_fixed_precision<cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3> >::value>::type
setup_karatsuba(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3>& b)
{
typedef cpp_int_backend<0, 0, signed_magnitude, unchecked, std::allocator<limb_type> > variable_precision_type;
variable_precision_type a_t(a.limbs(), 0, a.size()), b_t(b.limbs(), 0, b.size());
unsigned as = a.size();
unsigned bs = b.size();
unsigned s = as > bs ? as : bs;
unsigned sz = as + bs;
unsigned storage_size = karatsuba_storage_size(s);

if (sz * sizeof(limb_type) * CHAR_BIT <= MaxBits1)
{
result.resize(sz, sz);
variable_precision_type t(result.limbs(), 0, result.size());
typename variable_precision_type::scoped_shared_storage storage(t.allocator(), storage_size);
multiply_karatsuba(t, a_t, b_t, storage);
}
else
{
typename variable_precision_type::scoped_shared_storage storage(variable_precision_type::allocator_type(), sz + storage_size);
variable_precision_type t(storage, sz);
multiply_karatsuba(t, a_t, b_t, storage);
result = t;
}
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2, unsigned MinBits3, unsigned MaxBits3, cpp_integer_type SignType3, cpp_int_check_type Checked3, class Allocator3>
inline BOOST_MP_CXX14_CONSTEXPR void
eval_multiply_comba(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3>& b) BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value))
{
int as                                                                                         = a.size(),
bs                                                                                         = b.size(),
rs                                                                                         = result.size();
typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::limb_pointer pr = result.limbs();

double_limb_type carry    = 0,
temp     = 0;
limb_type      overflow   = 0;
const unsigned limb_bits  = sizeof(limb_type) * CHAR_BIT;
const bool     must_throw = rs < as + bs - 1;
for (int r = 0, lim = (std::min)(rs, as + bs - 1); r < lim; ++r, overflow = 0)
{
int i = r >= as ? as - 1 : r,
j = r - i,
k = i < bs - j ? i + 1 : bs - j; 

typename cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>::const_limb_pointer pa = a.limbs() + i;
typename cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3>::const_limb_pointer pb = b.limbs() + j;

temp = carry;
carry += static_cast<double_limb_type>(*(pa)) * (*(pb));
overflow += carry < temp;
for (--k; k; k--)
{
temp = carry;
carry += static_cast<double_limb_type>(*(--pa)) * (*(++pb));
overflow += carry < temp;
}
*(pr++) = static_cast<limb_type>(carry);
carry   = (static_cast<double_limb_type>(overflow) << limb_bits) | (carry >> limb_bits);
}
if (carry || must_throw)
{
resize_for_carry(result, as + bs);
if ((int)result.size() >= as + bs)
*pr = static_cast<limb_type>(carry);
}
}
template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2, unsigned MinBits3, unsigned MaxBits3, cpp_integer_type SignType3, cpp_int_check_type Checked3, class Allocator3>
inline BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3>& b) 
BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value 
&& (karatsuba_cutoff * sizeof(limb_type) * CHAR_BIT > MaxBits1) 
&& (karatsuba_cutoff * sizeof(limb_type)* CHAR_BIT > MaxBits2) 
&& (karatsuba_cutoff * sizeof(limb_type)* CHAR_BIT > MaxBits3)))
{
unsigned                                                                                          as = a.size();
unsigned                                                                                          bs = b.size();
typename cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>::const_limb_pointer pa = a.limbs();
typename cpp_int_backend<MinBits3, MaxBits3, SignType3, Checked3, Allocator3>::const_limb_pointer pb = b.limbs();
if (as == 1)
{
bool s = b.sign() != a.sign();
if (bs == 1)
{
result = static_cast<double_limb_type>(*pa) * static_cast<double_limb_type>(*pb);
}
else
{
limb_type l = *pa;
eval_multiply(result, b, l);
}
result.sign(s);
return;
}
if (bs == 1)
{
bool      s = b.sign() != a.sign();
limb_type l = *pb;
eval_multiply(result, a, l);
result.sign(s);
return;
}

if ((void*)&result == (void*)&a)
{
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> t(a);
eval_multiply(result, t, b);
return;
}
if ((void*)&result == (void*)&b)
{
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> t(b);
eval_multiply(result, a, t);
return;
}

#ifdef BOOST_NO_CXX14_CONSTEXPR
static const double_limb_type limb_max        = ~static_cast<limb_type>(0u);
static const double_limb_type double_limb_max = ~static_cast<double_limb_type>(0u);
#else
constexpr const double_limb_type limb_max = ~static_cast<limb_type>(0u);
constexpr const double_limb_type double_limb_max = ~static_cast<double_limb_type>(0u);
#endif
result.resize(as + bs, as + bs - 1);
#ifndef BOOST_MP_NO_CONSTEXPR_DETECTION
if (!BOOST_MP_IS_CONST_EVALUATED(as) && (as >= karatsuba_cutoff && bs >= karatsuba_cutoff))
#else
if (as >= karatsuba_cutoff && bs >= karatsuba_cutoff)
#endif
{
setup_karatsuba(result, a, b);
result.sign(a.sign() != b.sign());
return;
}
typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::limb_pointer pr = result.limbs();
BOOST_STATIC_ASSERT(double_limb_max - 2 * limb_max >= limb_max * limb_max);

#ifndef BOOST_MP_NO_CONSTEXPR_DETECTION
if (BOOST_MP_IS_CONST_EVALUATED(as))
{
for (unsigned i = 0; i < result.size(); ++i)
pr[i] = 0;
}
else
#endif
std::memset(pr, 0, result.size() * sizeof(limb_type));   

#if defined(BOOST_MP_COMBA)
eval_multiply_comba(result, a, b);
#else

double_limb_type carry = 0;
for (unsigned i = 0; i < as; ++i)
{
BOOST_ASSERT(result.size() > i);
unsigned inner_limit = !is_fixed_precision<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value ? bs : (std::min)(result.size() - i, bs);
unsigned j           = 0;
for (; j < inner_limit; ++j)
{
BOOST_ASSERT(i + j < result.size());
#if (!defined(__GLIBCXX__) && !defined(__GLIBCPP__)) || !BOOST_WORKAROUND(BOOST_GCC_VERSION, <= 50100)
BOOST_ASSERT(!std::numeric_limits<double_limb_type>::is_specialized || ((std::numeric_limits<double_limb_type>::max)() - carry >
static_cast<double_limb_type>(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::max_limb_value) * static_cast<double_limb_type>(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::max_limb_value)));
#endif
carry += static_cast<double_limb_type>(pa[i]) * static_cast<double_limb_type>(pb[j]);
BOOST_ASSERT(!std::numeric_limits<double_limb_type>::is_specialized || ((std::numeric_limits<double_limb_type>::max)() - carry >= pr[i + j]));
carry += pr[i + j];
#ifdef __MSVC_RUNTIME_CHECKS
pr[i + j] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
#else
pr[i + j] = static_cast<limb_type>(carry);
#endif
carry >>= cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::limb_bits;
BOOST_ASSERT(carry <= (cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::max_limb_value));
}
if (carry)
{
resize_for_carry(result, i + j + 1); 
if (i + j < result.size())
#ifdef __MSVC_RUNTIME_CHECKS
pr[i + j] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
#else
pr[i + j] = static_cast<limb_type>(carry);
#endif
}
carry = 0;
}
#endif 

result.normalize();
result.sign(a.sign() != b.sign());
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a) 
BOOST_MP_NOEXCEPT_IF((noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>&>()))))
{
eval_multiply(result, result, a);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result, const limb_type& val) 
BOOST_MP_NOEXCEPT_IF((noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const limb_type&>()))))
{
eval_multiply(result, result, val);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const double_limb_type&                                                     val) 
BOOST_MP_NOEXCEPT_IF(
(noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>&>(), std::declval<const limb_type&>())))
&& (noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>())))
)
{
if (val <= (std::numeric_limits<limb_type>::max)())
{
eval_multiply(result, a, static_cast<limb_type>(val));
}
else
{
#if BOOST_ENDIAN_LITTLE_BYTE && !defined(BOOST_MP_TEST_NO_LE)
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> t(val);
#else
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> t;
t = val;
#endif
eval_multiply(result, a, t);
}
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result, const double_limb_type& val)
BOOST_MP_NOEXCEPT_IF((noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const double_limb_type&>()))))
{
eval_multiply(result, result, val);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const signed_limb_type&                                                     val) 
BOOST_MP_NOEXCEPT_IF((noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>&>(), std::declval<const limb_type&>()))))
{
if (val > 0)
eval_multiply(result, a, static_cast<limb_type>(val));
else
{
eval_multiply(result, a, static_cast<limb_type>(boost::multiprecision::detail::unsigned_abs(val)));
result.negate();
}
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result, const signed_limb_type& val)
BOOST_MP_NOEXCEPT_IF((noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const limb_type&>()))))
{
eval_multiply(result, result, val);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2>
inline BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && !is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>& a,
const signed_double_limb_type&                                              val)
BOOST_MP_NOEXCEPT_IF(
(noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>&>(), std::declval<const limb_type&>())))
&& (noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>())))
)
{
if (val > 0)
{
if (val <= (std::numeric_limits<limb_type>::max)())
{
eval_multiply(result, a, static_cast<limb_type>(val));
return;
}
}
else if (val >= -static_cast<signed_double_limb_type>((std::numeric_limits<limb_type>::max)()))
{
eval_multiply(result, a, static_cast<limb_type>(boost::multiprecision::detail::unsigned_abs(val)));
result.negate();
return;
}
#if BOOST_ENDIAN_LITTLE_BYTE && !defined(BOOST_MP_TEST_NO_LE)
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> t(val);
#else
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> t;
t = val;
#endif
eval_multiply(result, a, t);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result, const signed_double_limb_type& val)
BOOST_MP_NOEXCEPT_IF(
(noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const limb_type&>())))
&& (noexcept(eval_multiply(std::declval<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>(), std::declval<const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&>())))
)
{
eval_multiply(result, result, val);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && (is_signed_number<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value || is_signed_number<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value)>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& o) BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value))
{
*result.limbs() = detail::checked_multiply(*result.limbs(), *o.limbs(), typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::checked_type());
result.sign(result.sign() != o.sign());
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && is_unsigned_number<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& o) BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value))
{
*result.limbs() = detail::checked_multiply(*result.limbs(), *o.limbs(), typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::checked_type());
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && (is_signed_number<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value || is_signed_number<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value)>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& a,
const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& b) BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value))
{
*result.limbs() = detail::checked_multiply(*a.limbs(), *b.limbs(), typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::checked_type());
result.sign(a.sign() != b.sign());
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && is_unsigned_number<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& a,
const cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& b) BOOST_MP_NOEXCEPT_IF((is_non_throwing_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value))
{
*result.limbs() = detail::checked_multiply(*a.limbs(), *b.limbs(), typename cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>::checked_type());
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result,
signed_double_limb_type a, signed_double_limb_type b)
{
#ifdef BOOST_NO_CXX14_CONSTEXPR
static const signed_double_limb_type mask      = ~static_cast<limb_type>(0);
static const unsigned                limb_bits = sizeof(limb_type) * CHAR_BIT;
#else
constexpr const signed_double_limb_type mask = ~static_cast<limb_type>(0);
constexpr const unsigned limb_bits = sizeof(limb_type) * CHAR_BIT;
#endif
bool s = false;
if (a < 0)
{
a = -a;
s = true;
}
if (b < 0)
{
b = -b;
s = !s;
}
double_limb_type w = a & mask;
double_limb_type x = a >> limb_bits;
double_limb_type y = b & mask;
double_limb_type z = b >> limb_bits;

result.resize(4, 4);
limb_type* pr = result.limbs();

double_limb_type carry = w * y;
#ifdef __MSVC_RUNTIME_CHECKS
pr[0] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
carry >>= limb_bits;
carry += w * z + x * y;
pr[1] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
carry >>= limb_bits;
carry += x * z;
pr[2] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
pr[3] = static_cast<limb_type>(carry >> limb_bits);
#else
pr[0] = static_cast<limb_type>(carry);
carry >>= limb_bits;
carry += w * z + x * y;
pr[1] = static_cast<limb_type>(carry);
carry >>= limb_bits;
carry += x * z;
pr[2] = static_cast<limb_type>(carry);
pr[3] = static_cast<limb_type>(carry >> limb_bits);
#endif
result.sign(s);
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result,
double_limb_type a, double_limb_type b)
{
#ifdef BOOST_NO_CXX14_CONSTEXPR
static const signed_double_limb_type mask      = ~static_cast<limb_type>(0);
static const unsigned                limb_bits = sizeof(limb_type) * CHAR_BIT;
#else
constexpr const signed_double_limb_type mask = ~static_cast<limb_type>(0);
constexpr const unsigned limb_bits = sizeof(limb_type) * CHAR_BIT;
#endif

double_limb_type w = a & mask;
double_limb_type x = a >> limb_bits;
double_limb_type y = b & mask;
double_limb_type z = b >> limb_bits;

result.resize(4, 4);
limb_type* pr = result.limbs();

double_limb_type carry = w * y;
#ifdef __MSVC_RUNTIME_CHECKS
pr[0] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
carry >>= limb_bits;
carry += w * z;
pr[1] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
carry >>= limb_bits;
pr[2] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
carry = x * y + pr[1];
pr[1] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
carry >>= limb_bits;
carry += pr[2] + x * z;
pr[2] = static_cast<limb_type>(carry & ~static_cast<limb_type>(0));
pr[3] = static_cast<limb_type>(carry >> limb_bits);
#else
pr[0] = static_cast<limb_type>(carry);
carry >>= limb_bits;
carry += w * z;
pr[1] = static_cast<limb_type>(carry);
carry >>= limb_bits;
pr[2] = static_cast<limb_type>(carry);
carry = x * y + pr[1];
pr[1] = static_cast<limb_type>(carry);
carry >>= limb_bits;
carry += pr[2] + x * z;
pr[2] = static_cast<limb_type>(carry);
pr[3] = static_cast<limb_type>(carry >> limb_bits);
#endif
result.sign(false);
result.normalize();
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1,
unsigned MinBits2, unsigned MaxBits2, cpp_integer_type SignType2, cpp_int_check_type Checked2, class Allocator2>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<
!is_trivial_cpp_int<cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::value && is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value && is_trivial_cpp_int<cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> >::value>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>&       result,
cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> const& a,
cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2> const& b)
{
typedef typename boost::multiprecision::detail::canonical<typename cpp_int_backend<MinBits2, MaxBits2, SignType2, Checked2, Allocator2>::local_limb_type, cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1> >::type canonical_type;
eval_multiply(result, static_cast<canonical_type>(*a.limbs()), static_cast<canonical_type>(*b.limbs()));
result.sign(a.sign() != b.sign());
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, class SI>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_signed<SI>::value && (sizeof(SI) <= sizeof(signed_double_limb_type) / 2)>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result,
SI a, SI b)
{
result = static_cast<signed_double_limb_type>(a) * static_cast<signed_double_limb_type>(b);
}

template <unsigned MinBits1, unsigned MaxBits1, cpp_integer_type SignType1, cpp_int_check_type Checked1, class Allocator1, class UI>
BOOST_MP_FORCEINLINE BOOST_MP_CXX14_CONSTEXPR typename enable_if_c<is_unsigned<UI>::value && (sizeof(UI) <= sizeof(signed_double_limb_type) / 2)>::type
eval_multiply(
cpp_int_backend<MinBits1, MaxBits1, SignType1, Checked1, Allocator1>& result,
UI a, UI b)
{
result = static_cast<double_limb_type>(a) * static_cast<double_limb_type>(b);
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}}} 

#endif
