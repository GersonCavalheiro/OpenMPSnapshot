
#ifndef BOOST_MP_CPP_DEC_FLOAT_BACKEND_HPP
#define BOOST_MP_CPP_DEC_FLOAT_BACKEND_HPP

#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <limits>
#ifndef BOOST_NO_CXX11_HDR_ARRAY
#include <array>
#else
#include <boost/array.hpp>
#endif
#include <boost/cstdint.hpp>
#include <boost/functional/hash_fwd.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/detail/big_lanczos.hpp>
#include <boost/multiprecision/detail/dynamic_array.hpp>
#include <boost/multiprecision/detail/itos.hpp>

#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <boost/math/special_functions/acosh.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <boost/math/special_functions/cbrt.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/special_functions/gamma.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 6326) 
#endif

namespace boost {
namespace multiprecision {
namespace backends {

template <unsigned Digits10, class ExponentType = boost::int32_t, class Allocator = void>
class cpp_dec_float;

} 

template <unsigned Digits10, class ExponentType, class Allocator>
struct number_category<backends::cpp_dec_float<Digits10, ExponentType, Allocator> > : public mpl::int_<number_kind_floating_point>
{};

namespace backends {

template <unsigned Digits10, class ExponentType, class Allocator>
class cpp_dec_float
{
private:
static const boost::int32_t cpp_dec_float_digits10_setting = Digits10;

BOOST_STATIC_ASSERT_MSG(boost::is_signed<ExponentType>::value, "ExponentType must be a signed built in integer type.");
BOOST_STATIC_ASSERT_MSG(sizeof(ExponentType) > 1, "ExponentType is too small.");

public:
typedef mpl::list<boost::long_long_type>  signed_types;
typedef mpl::list<boost::ulong_long_type> unsigned_types;
typedef mpl::list<double, long double>    float_types;
typedef ExponentType                      exponent_type;

static const boost::int32_t cpp_dec_float_radix             = 10L;
static const boost::int32_t cpp_dec_float_digits10_limit_lo = 9L;
static const boost::int32_t cpp_dec_float_digits10_limit_hi = boost::integer_traits<boost::int32_t>::const_max - 100;
static const boost::int32_t cpp_dec_float_digits10          = ((cpp_dec_float_digits10_setting < cpp_dec_float_digits10_limit_lo) ? cpp_dec_float_digits10_limit_lo : ((cpp_dec_float_digits10_setting > cpp_dec_float_digits10_limit_hi) ? cpp_dec_float_digits10_limit_hi : cpp_dec_float_digits10_setting));
static const ExponentType   cpp_dec_float_max_exp10         = (static_cast<ExponentType>(1) << (std::numeric_limits<ExponentType>::digits - 5));
static const ExponentType   cpp_dec_float_min_exp10         = -cpp_dec_float_max_exp10;
static const ExponentType   cpp_dec_float_max_exp           = cpp_dec_float_max_exp10;
static const ExponentType   cpp_dec_float_min_exp           = cpp_dec_float_min_exp10;

BOOST_STATIC_ASSERT((cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_max_exp10 == -cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_min_exp10));

private:
static const boost::int32_t cpp_dec_float_elem_digits10 = 8L;
static const boost::int32_t cpp_dec_float_elem_mask     = 100000000L;

BOOST_STATIC_ASSERT(0 == cpp_dec_float_max_exp10 % cpp_dec_float_elem_digits10);

static const boost::int32_t cpp_dec_float_elem_number_request = static_cast<boost::int32_t>((cpp_dec_float_digits10 / cpp_dec_float_elem_digits10) + (((cpp_dec_float_digits10 % cpp_dec_float_elem_digits10) != 0) ? 1 : 0));

static const boost::int32_t cpp_dec_float_elem_number = static_cast<boost::int32_t>(((cpp_dec_float_elem_number_request < 2L) ? 2L : cpp_dec_float_elem_number_request) + 3L);

public:
static const boost::int32_t cpp_dec_float_total_digits10 = static_cast<boost::int32_t>(cpp_dec_float_elem_number * cpp_dec_float_elem_digits10);

private:
typedef enum enum_fpclass_type
{
cpp_dec_float_finite,
cpp_dec_float_inf,
cpp_dec_float_NaN
} fpclass_type;

#ifndef BOOST_NO_CXX11_HDR_ARRAY
typedef typename mpl::if_<is_void<Allocator>,
std::array<boost::uint32_t, cpp_dec_float_elem_number>,
detail::dynamic_array<boost::uint32_t, cpp_dec_float_elem_number, Allocator> >::type array_type;
#else
typedef typename mpl::if_<is_void<Allocator>,
boost::array<boost::uint32_t, cpp_dec_float_elem_number>,
detail::dynamic_array<boost::uint32_t, cpp_dec_float_elem_number, Allocator> >::type array_type;
#endif

array_type     data;
ExponentType   exp;
bool           neg;
fpclass_type   fpclass;
boost::int32_t prec_elem;

cpp_dec_float(fpclass_type c) : data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(c),
prec_elem(cpp_dec_float_elem_number) {}

struct initializer
{
initializer()
{
cpp_dec_float<Digits10, ExponentType, Allocator>::nan();
cpp_dec_float<Digits10, ExponentType, Allocator>::inf();
(cpp_dec_float<Digits10, ExponentType, Allocator>::min)();
(cpp_dec_float<Digits10, ExponentType, Allocator>::max)();
cpp_dec_float<Digits10, ExponentType, Allocator>::zero();
cpp_dec_float<Digits10, ExponentType, Allocator>::one();
cpp_dec_float<Digits10, ExponentType, Allocator>::two();
cpp_dec_float<Digits10, ExponentType, Allocator>::half();
cpp_dec_float<Digits10, ExponentType, Allocator>::double_min();
cpp_dec_float<Digits10, ExponentType, Allocator>::double_max();
cpp_dec_float<Digits10, ExponentType, Allocator>::long_long_max();
cpp_dec_float<Digits10, ExponentType, Allocator>::long_long_min();
cpp_dec_float<Digits10, ExponentType, Allocator>::ulong_long_max();
cpp_dec_float<Digits10, ExponentType, Allocator>::eps();
cpp_dec_float<Digits10, ExponentType, Allocator>::pow2(0);
}
void do_nothing() {}
};

static initializer init;

struct long_double_initializer
{
long_double_initializer()
{
cpp_dec_float<Digits10, ExponentType, Allocator>::long_double_max();
cpp_dec_float<Digits10, ExponentType, Allocator>::long_double_min();
}
void do_nothing() {}
};

static long_double_initializer linit;

public:
cpp_dec_float() BOOST_MP_NOEXCEPT_IF(noexcept(array_type())) : data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(cpp_dec_float_finite),
prec_elem(cpp_dec_float_elem_number) {}

cpp_dec_float(const char* s) : data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(cpp_dec_float_finite),
prec_elem(cpp_dec_float_elem_number)
{
*this = s;
}

template <class I>
cpp_dec_float(I i, typename enable_if<is_unsigned<I> >::type* = 0) : data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(cpp_dec_float_finite),
prec_elem(cpp_dec_float_elem_number)
{
from_unsigned_long_long(i);
}

template <class I>
cpp_dec_float(I i, typename enable_if<is_signed<I> >::type* = 0) : data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(cpp_dec_float_finite),
prec_elem(cpp_dec_float_elem_number)
{
if (i < 0)
{
from_unsigned_long_long(boost::multiprecision::detail::unsigned_abs(i));
negate();
}
else
from_unsigned_long_long(i);
}

cpp_dec_float(const cpp_dec_float& f) BOOST_MP_NOEXCEPT_IF(noexcept(array_type(std::declval<const array_type&>()))) : data(f.data),
exp(f.exp),
neg(f.neg),
fpclass(f.fpclass),
prec_elem(f.prec_elem) {}

template <unsigned D, class ET, class A>
cpp_dec_float(const cpp_dec_float<D, ET, A>& f, typename enable_if_c<D <= Digits10>::type* = 0) : data(),
exp(f.exp),
neg(f.neg),
fpclass(static_cast<fpclass_type>(static_cast<int>(f.fpclass))),
prec_elem(cpp_dec_float_elem_number)
{
std::copy(f.data.begin(), f.data.begin() + f.prec_elem, data.begin());
}
template <unsigned D, class ET, class A>
explicit cpp_dec_float(const cpp_dec_float<D, ET, A>& f, typename disable_if_c<D <= Digits10>::type* = 0) : data(),
exp(f.exp),
neg(f.neg),
fpclass(static_cast<fpclass_type>(static_cast<int>(f.fpclass))),
prec_elem(cpp_dec_float_elem_number)
{
std::copy(f.data.begin(), f.data.begin() + prec_elem, data.begin());
}

template <class F>
cpp_dec_float(const F val, typename enable_if_c<is_floating_point<F>::value
#ifdef BOOST_HAS_FLOAT128
&& !boost::is_same<F, __float128>::value
#endif
>::type* = 0) : data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(cpp_dec_float_finite),
prec_elem(cpp_dec_float_elem_number)
{
*this = val;
}

cpp_dec_float(const double mantissa, const ExponentType exponent);

std::size_t hash() const
{
std::size_t result = 0;
for (int i = 0; i < prec_elem; ++i)
boost::hash_combine(result, data[i]);
boost::hash_combine(result, exp);
boost::hash_combine(result, neg);
boost::hash_combine(result, fpclass);
return result;
}

static const cpp_dec_float& nan()
{
static const cpp_dec_float val(cpp_dec_float_NaN);
init.do_nothing();
return val;
}

static const cpp_dec_float& inf()
{
static const cpp_dec_float val(cpp_dec_float_inf);
init.do_nothing();
return val;
}

static const cpp_dec_float&(max)()
{
init.do_nothing();
static cpp_dec_float val_max = std::string("1.0e" + boost::multiprecision::detail::itos(cpp_dec_float_max_exp10)).c_str();
return val_max;
}

static const cpp_dec_float&(min)()
{
init.do_nothing();
static cpp_dec_float val_min = std::string("1.0e" + boost::multiprecision::detail::itos(cpp_dec_float_min_exp10)).c_str();
return val_min;
}

static const cpp_dec_float& zero()
{
init.do_nothing();
static cpp_dec_float val(static_cast<boost::ulong_long_type>(0u));
return val;
}

static const cpp_dec_float& one()
{
init.do_nothing();
static cpp_dec_float val(static_cast<boost::ulong_long_type>(1u));
return val;
}

static const cpp_dec_float& two()
{
init.do_nothing();
static cpp_dec_float val(static_cast<boost::ulong_long_type>(2u));
return val;
}

static const cpp_dec_float& half()
{
init.do_nothing();
static cpp_dec_float val(0.5L);
return val;
}

static const cpp_dec_float& double_min()
{
init.do_nothing();
static cpp_dec_float val((std::numeric_limits<double>::min)());
return val;
}

static const cpp_dec_float& double_max()
{
init.do_nothing();
static cpp_dec_float val((std::numeric_limits<double>::max)());
return val;
}

static const cpp_dec_float& long_double_min()
{
linit.do_nothing();
#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
static cpp_dec_float val(static_cast<long double>((std::numeric_limits<double>::min)()));
#else
static cpp_dec_float val((std::numeric_limits<long double>::min)());
#endif
return val;
}

static const cpp_dec_float& long_double_max()
{
linit.do_nothing();
#ifdef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
static cpp_dec_float val(static_cast<long double>((std::numeric_limits<double>::max)()));
#else
static cpp_dec_float val((std::numeric_limits<long double>::max)());
#endif
return val;
}

static const cpp_dec_float& long_long_max()
{
init.do_nothing();
static cpp_dec_float val((std::numeric_limits<boost::long_long_type>::max)());
return val;
}

static const cpp_dec_float& long_long_min()
{
init.do_nothing();
static cpp_dec_float val((std::numeric_limits<boost::long_long_type>::min)());
return val;
}

static const cpp_dec_float& ulong_long_max()
{
init.do_nothing();
static cpp_dec_float val((std::numeric_limits<boost::ulong_long_type>::max)());
return val;
}

static const cpp_dec_float& eps()
{
init.do_nothing();
static cpp_dec_float val(1.0, 1 - static_cast<int>(cpp_dec_float_digits10));
return val;
}

cpp_dec_float& operator=(const cpp_dec_float& v) BOOST_MP_NOEXCEPT_IF(noexcept(std::declval<array_type&>() = std::declval<const array_type&>()))
{
data      = v.data;
exp       = v.exp;
neg       = v.neg;
fpclass   = v.fpclass;
prec_elem = v.prec_elem;
return *this;
}

template <unsigned D>
cpp_dec_float& operator=(const cpp_dec_float<D>& f)
{
exp            = f.exp;
neg            = f.neg;
fpclass        = static_cast<enum_fpclass_type>(static_cast<int>(f.fpclass));
unsigned elems = (std::min)(f.prec_elem, cpp_dec_float_elem_number);
std::copy(f.data.begin(), f.data.begin() + elems, data.begin());
std::fill(data.begin() + elems, data.end(), 0);
prec_elem = cpp_dec_float_elem_number;
return *this;
}

cpp_dec_float& operator=(boost::long_long_type v)
{
if (v < 0)
{
from_unsigned_long_long(1u - boost::ulong_long_type(v + 1)); 
negate();
}
else
from_unsigned_long_long(v);
return *this;
}

cpp_dec_float& operator=(boost::ulong_long_type v)
{
from_unsigned_long_long(v);
return *this;
}

template <class Float>
typename boost::enable_if_c<boost::is_floating_point<Float>::value, cpp_dec_float&>::type operator=(Float v);

cpp_dec_float& operator=(const char* v)
{
rd_string(v);
return *this;
}

cpp_dec_float& operator+=(const cpp_dec_float& v);
cpp_dec_float& operator-=(const cpp_dec_float& v);
cpp_dec_float& operator*=(const cpp_dec_float& v);
cpp_dec_float& operator/=(const cpp_dec_float& v);

cpp_dec_float& add_unsigned_long_long(const boost::ulong_long_type n)
{
cpp_dec_float t;
t.from_unsigned_long_long(n);
return *this += t;
}

cpp_dec_float& sub_unsigned_long_long(const boost::ulong_long_type n)
{
cpp_dec_float t;
t.from_unsigned_long_long(n);
return *this -= t;
}

cpp_dec_float& mul_unsigned_long_long(const boost::ulong_long_type n);
cpp_dec_float& div_unsigned_long_long(const boost::ulong_long_type n);

cpp_dec_float& calculate_inv();
cpp_dec_float& calculate_sqrt();

void negate()
{
if (!iszero())
neg = !neg;
}

bool isnan BOOST_PREVENT_MACRO_SUBSTITUTION() const { return (fpclass == cpp_dec_float_NaN); }
bool isinf BOOST_PREVENT_MACRO_SUBSTITUTION() const { return (fpclass == cpp_dec_float_inf); }
bool isfinite BOOST_PREVENT_MACRO_SUBSTITUTION() const { return (fpclass == cpp_dec_float_finite); }

bool iszero() const
{
return ((fpclass == cpp_dec_float_finite) && (data[0u] == 0u));
}

bool isone() const;
bool isint() const;
bool isneg() const { return neg; }

cpp_dec_float& operator++()
{
return *this += one();
}

cpp_dec_float& operator--()
{
return *this -= one();
}

std::string str(boost::intmax_t digits, std::ios_base::fmtflags f) const;

int compare(const cpp_dec_float& v) const;

template <class V>
int compare(const V& v) const
{
cpp_dec_float<Digits10, ExponentType, Allocator> t;
t = v;
return compare(t);
}

void swap(cpp_dec_float& v)
{
data.swap(v.data);
std::swap(exp, v.exp);
std::swap(neg, v.neg);
std::swap(fpclass, v.fpclass);
std::swap(prec_elem, v.prec_elem);
}

double                 extract_double() const;
long double            extract_long_double() const;
boost::long_long_type  extract_signed_long_long() const;
boost::ulong_long_type extract_unsigned_long_long() const;
void                   extract_parts(double& mantissa, ExponentType& exponent) const;
cpp_dec_float          extract_integer_part() const;

void precision(const boost::int32_t prec_digits)
{
if (prec_digits >= cpp_dec_float_total_digits10)
{
prec_elem = cpp_dec_float_elem_number;
}
else
{
const boost::int32_t elems = static_cast<boost::int32_t>(static_cast<boost::int32_t>((prec_digits + (cpp_dec_float_elem_digits10 / 2)) / cpp_dec_float_elem_digits10) + static_cast<boost::int32_t>(((prec_digits % cpp_dec_float_elem_digits10) != 0) ? 1 : 0));

prec_elem = (std::min)(cpp_dec_float_elem_number, (std::max)(elems, static_cast<boost::int32_t>(2)));
}
}
static cpp_dec_float pow2(boost::long_long_type i);
ExponentType         order() const
{
const bool bo_order_is_zero = ((!(isfinite)()) || (data[0] == static_cast<boost::uint32_t>(0u)));
ExponentType prefix = 0;

if (data[0] >= 100000UL)
{
if (data[0] >= 10000000UL)
{
if (data[0] >= 100000000UL)
{
if (data[0] >= 1000000000UL)
prefix = 9;
else
prefix = 8;
}
else
prefix = 7;
}
else
{
if (data[0] >= 1000000UL)
prefix = 6;
else
prefix = 5;
}
}
else
{
if (data[0] >= 1000UL)
{
if (data[0] >= 10000UL)
prefix = 4;
else
prefix = 3;
}
else
{
if (data[0] >= 100)
prefix = 2;
else if (data[0] >= 10)
prefix = 1;
}
}

return (bo_order_is_zero ? static_cast<ExponentType>(0) : static_cast<ExponentType>(exp + prefix));
}

template <class Archive>
void serialize(Archive& ar, const unsigned int )
{
for (unsigned i = 0; i < data.size(); ++i)
ar& boost::make_nvp("digit", data[i]);
ar& boost::make_nvp("exponent", exp);
ar& boost::make_nvp("sign", neg);
ar& boost::make_nvp("class-type", fpclass);
ar& boost::make_nvp("precision", prec_elem);
}

private:
static bool data_elem_is_non_zero_predicate(const boost::uint32_t& d) { return (d != static_cast<boost::uint32_t>(0u)); }
static bool data_elem_is_non_nine_predicate(const boost::uint32_t& d) { return (d != static_cast<boost::uint32_t>(cpp_dec_float::cpp_dec_float_elem_mask - 1)); }
static bool char_is_nonzero_predicate(const char& c) { return (c != static_cast<char>('0')); }

void from_unsigned_long_long(const boost::ulong_long_type u);

int cmp_data(const array_type& vd) const;

static boost::uint32_t mul_loop_uv(boost::uint32_t* const u, const boost::uint32_t* const v, const boost::int32_t p);
static boost::uint32_t mul_loop_n(boost::uint32_t* const u, boost::uint32_t n, const boost::int32_t p);
static boost::uint32_t div_loop_n(boost::uint32_t* const u, boost::uint32_t n, const boost::int32_t p);

bool rd_string(const char* const s);

template <unsigned D, class ET, class A>
friend class cpp_dec_float;
};

template <unsigned Digits10, class ExponentType, class Allocator>
typename cpp_dec_float<Digits10, ExponentType, Allocator>::initializer cpp_dec_float<Digits10, ExponentType, Allocator>::init;
template <unsigned Digits10, class ExponentType, class Allocator>
typename cpp_dec_float<Digits10, ExponentType, Allocator>::long_double_initializer cpp_dec_float<Digits10, ExponentType, Allocator>::linit;

template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_radix;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10_setting;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10_limit_lo;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10_limit_hi;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10;
template <unsigned Digits10, class ExponentType, class Allocator>
const ExponentType cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_max_exp;
template <unsigned Digits10, class ExponentType, class Allocator>
const ExponentType cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_min_exp;
template <unsigned Digits10, class ExponentType, class Allocator>
const ExponentType cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_max_exp10;
template <unsigned Digits10, class ExponentType, class Allocator>
const ExponentType cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_min_exp10;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_elem_digits10;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_elem_number_request;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_elem_number;
template <unsigned Digits10, class ExponentType, class Allocator>
const boost::int32_t cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_elem_mask;

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::operator+=(const cpp_dec_float<Digits10, ExponentType, Allocator>& v)
{
if ((isnan)())
{
return *this;
}

if ((isinf)())
{
if ((v.isinf)() && (isneg() != v.isneg()))
{
*this = nan();
}
return *this;
}

if (iszero())
{
return operator=(v);
}

if ((v.isnan)() || (v.isinf)())
{
*this = v;
return *this;
}

static const ExponentType max_delta_exp = static_cast<ExponentType>((cpp_dec_float_elem_number - 1) * cpp_dec_float_elem_digits10);

const ExponentType ofs_exp = static_cast<ExponentType>(exp - v.exp);

if (v.iszero() || (ofs_exp > max_delta_exp))
{
return *this;
}
else if (ofs_exp < -max_delta_exp)
{
return operator=(v);
}


typename array_type::iterator       p_u    = data.begin();
typename array_type::const_iterator p_v    = v.data.begin();
bool                                b_copy = false;
const boost::int32_t                ofs    = static_cast<boost::int32_t>(static_cast<boost::int32_t>(ofs_exp) / cpp_dec_float_elem_digits10);
array_type                          n_data;

if (neg == v.neg)
{
if (ofs >= static_cast<boost::int32_t>(0))
{
std::copy(v.data.begin(), v.data.end() - static_cast<size_t>(ofs), n_data.begin() + static_cast<size_t>(ofs));
std::fill(n_data.begin(), n_data.begin() + static_cast<size_t>(ofs), static_cast<boost::uint32_t>(0u));
p_v = n_data.begin();
}
else
{
std::copy(data.begin(), data.end() - static_cast<size_t>(-ofs), n_data.begin() + static_cast<size_t>(-ofs));
std::fill(n_data.begin(), n_data.begin() + static_cast<size_t>(-ofs), static_cast<boost::uint32_t>(0u));
p_u    = n_data.begin();
b_copy = true;
}

boost::uint32_t carry = static_cast<boost::uint32_t>(0u);

for (boost::int32_t j = static_cast<boost::int32_t>(cpp_dec_float_elem_number - static_cast<boost::int32_t>(1)); j >= static_cast<boost::int32_t>(0); j--)
{
boost::uint32_t t = static_cast<boost::uint32_t>(static_cast<boost::uint32_t>(p_u[j] + p_v[j]) + carry);
carry             = t / static_cast<boost::uint32_t>(cpp_dec_float_elem_mask);
p_u[j]            = static_cast<boost::uint32_t>(t - static_cast<boost::uint32_t>(carry * static_cast<boost::uint32_t>(cpp_dec_float_elem_mask)));
}

if (b_copy)
{
data = n_data;
exp  = v.exp;
}

if (carry != static_cast<boost::uint32_t>(0u))
{
std::copy_backward(data.begin(), data.end() - static_cast<std::size_t>(1u), data.end());
data[0] = carry;
exp += static_cast<ExponentType>(cpp_dec_float_elem_digits10);
}
}
else
{
if ((ofs > static_cast<boost::int32_t>(0)) || ((ofs == static_cast<boost::int32_t>(0)) && (cmp_data(v.data) > static_cast<boost::int32_t>(0))))
{
std::copy(v.data.begin(), v.data.end() - static_cast<size_t>(ofs), n_data.begin() + static_cast<size_t>(ofs));
std::fill(n_data.begin(), n_data.begin() + static_cast<size_t>(ofs), static_cast<boost::uint32_t>(0u));
p_v = n_data.begin();
}
else
{
if (ofs != static_cast<boost::int32_t>(0))
{
std::copy_backward(data.begin(), data.end() - static_cast<size_t>(-ofs), data.end());
std::fill(data.begin(), data.begin() + static_cast<size_t>(-ofs), static_cast<boost::uint32_t>(0u));
}

n_data = v.data;
p_u    = n_data.begin();
p_v    = data.begin();
b_copy = true;
}

boost::int32_t j;

boost::int32_t borrow = static_cast<boost::int32_t>(0);

for (j = static_cast<boost::int32_t>(cpp_dec_float_elem_number - static_cast<boost::int32_t>(1)); j >= static_cast<boost::int32_t>(0); j--)
{
boost::int32_t t = static_cast<boost::int32_t>(static_cast<boost::int32_t>(static_cast<boost::int32_t>(p_u[j]) - static_cast<boost::int32_t>(p_v[j])) - borrow);

if (t < static_cast<boost::int32_t>(0))
{
t += static_cast<boost::int32_t>(cpp_dec_float_elem_mask);
borrow = static_cast<boost::int32_t>(1);
}
else
{
borrow = static_cast<boost::int32_t>(0);
}

p_u[j] = static_cast<boost::uint32_t>(static_cast<boost::uint32_t>(t) % static_cast<boost::uint32_t>(cpp_dec_float_elem_mask));
}

if (b_copy)
{
data = n_data;
exp  = v.exp;
neg  = v.neg;
}

const typename array_type::const_iterator first_nonzero_elem = std::find_if(data.begin(), data.end(), data_elem_is_non_zero_predicate);

if (first_nonzero_elem != data.begin())
{
if (first_nonzero_elem == data.end())
{
neg = false;
exp = static_cast<ExponentType>(0);
}
else
{
const std::size_t sj = static_cast<std::size_t>(std::distance<typename array_type::const_iterator>(data.begin(), first_nonzero_elem));

std::copy(data.begin() + static_cast<std::size_t>(sj), data.end(), data.begin());
std::fill(data.end() - sj, data.end(), static_cast<boost::uint32_t>(0u));

exp -= static_cast<ExponentType>(sj * static_cast<std::size_t>(cpp_dec_float_elem_digits10));
}
}
}

if (iszero())
return (*this = zero());

const bool b_result_might_overflow = (exp >= static_cast<ExponentType>(cpp_dec_float_max_exp10));

if (b_result_might_overflow)
{
const bool b_result_is_neg = neg;
neg                        = false;

if (compare((cpp_dec_float::max)()) > 0)
*this = inf();

neg = b_result_is_neg;
}

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::operator-=(const cpp_dec_float<Digits10, ExponentType, Allocator>& v)
{
negate();
*this += v;
negate();
return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::operator*=(const cpp_dec_float<Digits10, ExponentType, Allocator>& v)
{
const bool b_result_is_neg = (neg != v.neg);

neg = false;

const bool b_u_is_inf  = (isinf)();
const bool b_v_is_inf  = (v.isinf)();
const bool b_u_is_zero = iszero();
const bool b_v_is_zero = v.iszero();

if (((isnan)() || (v.isnan)()) || (b_u_is_inf && b_v_is_zero) || (b_v_is_inf && b_u_is_zero))
{
*this = nan();
return *this;
}

if (b_u_is_inf || b_v_is_inf)
{
*this = inf();
if (b_result_is_neg)
negate();
return *this;
}

if (b_u_is_zero || b_v_is_zero)
{
return *this = zero();
}

const bool b_result_might_overflow  = ((exp + v.exp) >= static_cast<ExponentType>(cpp_dec_float_max_exp10));
const bool b_result_might_underflow = ((exp + v.exp) <= static_cast<ExponentType>(cpp_dec_float_min_exp10));

exp += v.exp;

const boost::int32_t prec_mul = (std::min)(prec_elem, v.prec_elem);

const boost::uint32_t carry = mul_loop_uv(data.data(), v.data.data(), prec_mul);

if (carry != static_cast<boost::uint32_t>(0u))
{
exp += cpp_dec_float_elem_digits10;

std::copy_backward(data.begin(),
data.begin() + static_cast<std::size_t>(prec_elem - static_cast<boost::int32_t>(1)),
data.begin() + static_cast<std::size_t>(prec_elem));

data.front() = carry;
}

if (b_result_might_overflow && (compare((cpp_dec_float::max)()) > 0))
{
*this = inf();
}

if (b_result_might_underflow && (compare((cpp_dec_float::min)()) < 0))
{
*this = zero();

return *this;
}

neg = b_result_is_neg;

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::operator/=(const cpp_dec_float<Digits10, ExponentType, Allocator>& v)
{
if (iszero())
{
if ((v.isnan)())
{
return *this = v;
}
else if (v.iszero())
{
return *this = nan();
}
}

const bool u_and_v_are_finite_and_identical = ((isfinite)() && (fpclass == v.fpclass) && (exp == v.exp) && (cmp_data(v.data) == static_cast<boost::int32_t>(0)));

if (u_and_v_are_finite_and_identical)
{
if (neg != v.neg)
{
*this = one();
negate();
}
else
*this = one();
return *this;
}
else
{
cpp_dec_float t(v);
t.calculate_inv();
return operator*=(t);
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::mul_unsigned_long_long(const boost::ulong_long_type n)
{

const bool b_neg = neg;

neg = false;

const bool b_u_is_inf  = (isinf)();
const bool b_n_is_zero = (n == static_cast<boost::int32_t>(0));

if ((isnan)() || (b_u_is_inf && b_n_is_zero))
{
return (*this = nan());
}

if (b_u_is_inf)
{
*this = inf();
if (b_neg)
negate();
return *this;
}

if (iszero() || b_n_is_zero)
{
return *this = zero();
}

if (n >= static_cast<boost::ulong_long_type>(cpp_dec_float_elem_mask))
{
neg = b_neg;
cpp_dec_float t;
t = n;
return operator*=(t);
}

if (n == static_cast<boost::ulong_long_type>(1u))
{
neg = b_neg;
return *this;
}

const boost::uint32_t nn    = static_cast<boost::uint32_t>(n);
const boost::uint32_t carry = mul_loop_n(data.data(), nn, prec_elem);

if (carry != static_cast<boost::uint32_t>(0u))
{
exp += static_cast<ExponentType>(cpp_dec_float_elem_digits10);

std::copy_backward(data.begin(),
data.begin() + static_cast<std::size_t>(prec_elem - static_cast<boost::int32_t>(1)),
data.begin() + static_cast<std::size_t>(prec_elem));

data.front() = static_cast<boost::uint32_t>(carry);
}

const bool b_result_might_overflow = (exp >= cpp_dec_float_max_exp10);

if (b_result_might_overflow && (compare((cpp_dec_float::max)()) > 0))
{
*this = inf();
}

neg = b_neg;

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::div_unsigned_long_long(const boost::ulong_long_type n)
{

const bool b_neg = neg;

neg = false;

if ((isnan)())
{
return *this;
}

if ((isinf)())
{
*this = inf();
if (b_neg)
negate();
return *this;
}

if (n == static_cast<boost::ulong_long_type>(0u))
{
if (iszero())
{
*this = nan();
return *this;
}
else
{
*this = inf();
if (isneg())
negate();
return *this;
}
}

if (iszero())
{
return *this;
}

if (n >= static_cast<boost::ulong_long_type>(cpp_dec_float_elem_mask))
{
neg = b_neg;
cpp_dec_float t;
t = n;
return operator/=(t);
}

const boost::uint32_t nn = static_cast<boost::uint32_t>(n);

if (nn > static_cast<boost::uint32_t>(1u))
{
const boost::uint32_t prev = div_loop_n(data.data(), nn, prec_elem);

if (data[0] == static_cast<boost::uint32_t>(0u))
{
exp -= static_cast<ExponentType>(cpp_dec_float_elem_digits10);

std::copy(data.begin() + static_cast<std::size_t>(1u),
data.begin() + static_cast<std::size_t>(prec_elem - static_cast<boost::int32_t>(1)),
data.begin());

data[prec_elem - static_cast<boost::int32_t>(1)] = static_cast<boost::uint32_t>(static_cast<boost::uint64_t>(prev * static_cast<boost::uint64_t>(cpp_dec_float_elem_mask)) / nn);
}
}

const bool b_result_might_underflow = (exp <= cpp_dec_float_min_exp10);

if (b_result_might_underflow && (compare((cpp_dec_float::min)()) < 0))
return (*this = zero());

neg = b_neg;

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::calculate_inv()
{
const bool b_neg = neg;

neg = false;

if (iszero())
{
*this = inf();
if (b_neg)
negate();
return *this;
}

if ((isnan)())
{
return *this;
}

if ((isinf)())
{
return *this = zero();
}

if (isone())
{
if (b_neg)
negate();
return *this;
}

cpp_dec_float<Digits10, ExponentType, Allocator> x(*this);

double       dd;
ExponentType ne;
x.extract_parts(dd, ne);

operator=(cpp_dec_float<Digits10, ExponentType, Allocator>(1.0 / dd, -ne));


static const boost::int32_t double_digits10_minus_a_few = std::numeric_limits<double>::digits10 - 3;

for (boost::int32_t digits = double_digits10_minus_a_few; digits <= cpp_dec_float_total_digits10; digits *= static_cast<boost::int32_t>(2))
{
precision(static_cast<boost::int32_t>((digits + 10) * static_cast<boost::int32_t>(2)));
x.precision(static_cast<boost::int32_t>((digits + 10) * static_cast<boost::int32_t>(2)));

cpp_dec_float t(*this);
t *= x;
t -= two();
t.negate();
*this *= t;
}

neg = b_neg;

prec_elem = cpp_dec_float_elem_number;

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>& cpp_dec_float<Digits10, ExponentType, Allocator>::calculate_sqrt()
{

if ((isinf)() && !isneg())
{
return *this;
}

if (isneg() || (!(isfinite)()))
{
*this = nan();
errno = EDOM;
return *this;
}

if (iszero() || isone())
{
return *this;
}

cpp_dec_float<Digits10, ExponentType, Allocator> x(*this);

double       dd;
ExponentType ne;
extract_parts(dd, ne);

if ((ne % static_cast<ExponentType>(2)) != static_cast<ExponentType>(0))
{
++ne;
dd /= 10.0;
}

const double sqd = std::sqrt(dd);

*this = cpp_dec_float<Digits10, ExponentType, Allocator>(sqd, static_cast<ExponentType>(ne / static_cast<ExponentType>(2)));

cpp_dec_float<Digits10, ExponentType, Allocator> vi(0.5 / sqd, static_cast<ExponentType>(-ne / static_cast<ExponentType>(2)));


static const boost::uint32_t double_digits10_minus_a_few = std::numeric_limits<double>::digits10 - 3;

for (boost::int32_t digits = double_digits10_minus_a_few; digits <= cpp_dec_float_total_digits10; digits *= 2u)
{
precision((digits + 10) * 2);
vi.precision((digits + 10) * 2);

cpp_dec_float t(*this);
t *= vi;
t.negate();
t.mul_unsigned_long_long(2u);
t += one();
t *= vi;
vi += t;

t = *this;
t *= *this;
t.negate();
t += x;
t *= vi;
*this += t;
}

prec_elem = cpp_dec_float_elem_number;

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
int cpp_dec_float<Digits10, ExponentType, Allocator>::cmp_data(const array_type& vd) const
{

const std::pair<typename array_type::const_iterator, typename array_type::const_iterator> mismatch_pair = std::mismatch(data.begin(), data.end(), vd.begin());

const bool is_equal = ((mismatch_pair.first == data.end()) && (mismatch_pair.second == vd.end()));

if (is_equal)
{
return 0;
}
else
{
return ((*mismatch_pair.first > *mismatch_pair.second) ? 1 : -1);
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
int cpp_dec_float<Digits10, ExponentType, Allocator>::compare(const cpp_dec_float& v) const
{

if ((!(isfinite)()) || (!(v.isfinite)()))
{
if ((isnan)() || (v.isnan)())
{
return ((isnan)() ? 1 : -1);
}

if ((isinf)() && (v.isinf)())
{
return ((neg == v.neg) ? 0 : (neg ? -1 : 1));
}

if ((isinf)())
{
return (isneg() ? -1 : 1);
}
else
{
return (v.neg ? 1 : -1);
}
}

if (iszero())
{
return (v.iszero() ? 0
: (v.neg ? 1 : -1));
}
else if (v.iszero())
{
return (neg ? -1 : 1);
}
else
{

if (neg != v.neg)
{
return (neg ? -1 : 1);
}
else if (exp != v.exp)
{
const int val_cexpression = ((exp < v.exp) ? 1 : -1);

return (neg ? val_cexpression : -val_cexpression);
}
else
{
const int val_cmp_data = cmp_data(v.data);

return ((!neg) ? val_cmp_data : -val_cmp_data);
}
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
bool cpp_dec_float<Digits10, ExponentType, Allocator>::isone() const
{

const bool not_negative_and_is_finite = ((!neg) && (isfinite)());

if (not_negative_and_is_finite)
{
if ((data[0u] == static_cast<boost::uint32_t>(1u)) && (exp == static_cast<ExponentType>(0)))
{
const typename array_type::const_iterator it_non_zero = std::find_if(data.begin(), data.end(), data_elem_is_non_zero_predicate);
return (it_non_zero == data.end());
}
else if ((data[0u] == static_cast<boost::uint32_t>(cpp_dec_float_elem_mask - 1)) && (exp == static_cast<ExponentType>(-cpp_dec_float_elem_digits10)))
{
const typename array_type::const_iterator it_non_nine = std::find_if(data.begin(), data.end(), data_elem_is_non_nine_predicate);
return (it_non_nine == data.end());
}
}

return false;
}

template <unsigned Digits10, class ExponentType, class Allocator>
bool cpp_dec_float<Digits10, ExponentType, Allocator>::isint() const
{
if (fpclass != cpp_dec_float_finite)
{
return false;
}

if (iszero())
{
return true;
}

if (exp < static_cast<ExponentType>(0))
{
return false;
} 

const typename array_type::size_type offset_decimal_part = static_cast<typename array_type::size_type>(exp / cpp_dec_float_elem_digits10) + 1u;

if (offset_decimal_part >= static_cast<typename array_type::size_type>(cpp_dec_float_elem_number))
{
return true;
}

typename array_type::const_iterator it_non_zero = std::find_if(data.begin() + offset_decimal_part, data.end(), data_elem_is_non_zero_predicate);

return (it_non_zero == data.end());
}

template <unsigned Digits10, class ExponentType, class Allocator>
void cpp_dec_float<Digits10, ExponentType, Allocator>::extract_parts(double& mantissa, ExponentType& exponent) const
{

exponent = exp;

boost::uint32_t p10  = static_cast<boost::uint32_t>(1u);
boost::uint32_t test = data[0u];

for (;;)
{
test /= static_cast<boost::uint32_t>(10u);

if (test == static_cast<boost::uint32_t>(0u))
{
break;
}

p10 *= static_cast<boost::uint32_t>(10u);
++exponent;
}

const int max_elem_in_double_count = static_cast<int>(static_cast<boost::int32_t>(std::numeric_limits<double>::digits10) / cpp_dec_float_elem_digits10) + (static_cast<int>(static_cast<boost::int32_t>(std::numeric_limits<double>::digits10) % cpp_dec_float_elem_digits10) != 0 ? 1 : 0) + 1;

const std::size_t max_elem_extract_count = static_cast<std::size_t>((std::min)(static_cast<boost::int32_t>(max_elem_in_double_count), cpp_dec_float_elem_number));

mantissa     = static_cast<double>(data[0]);
double scale = 1.0;

for (std::size_t i = 1u; i < max_elem_extract_count; i++)
{
scale /= static_cast<double>(cpp_dec_float_elem_mask);
mantissa += (static_cast<double>(data[i]) * scale);
}

mantissa /= static_cast<double>(p10);

if (neg)
{
mantissa = -mantissa;
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
double cpp_dec_float<Digits10, ExponentType, Allocator>::extract_double() const
{

if (!(isfinite)())
{
if ((isnan)())
{
return std::numeric_limits<double>::quiet_NaN();
}
else
{
return ((!neg) ? std::numeric_limits<double>::infinity()
: -std::numeric_limits<double>::infinity());
}
}

cpp_dec_float<Digits10, ExponentType, Allocator> xx(*this);
if (xx.isneg())
xx.negate();

if (iszero() || (xx.compare(double_min()) < 0))
{
return 0.0;
}

if (xx.compare(double_max()) > 0)
{
return ((!neg) ? std::numeric_limits<double>::infinity()
: -std::numeric_limits<double>::infinity());
}

std::stringstream ss;
ss.imbue(std::locale::classic());

ss << str(std::numeric_limits<double>::digits10 + (2 + 1), std::ios_base::scientific);

double d;
ss >> d;

return d;
}

template <unsigned Digits10, class ExponentType, class Allocator>
long double cpp_dec_float<Digits10, ExponentType, Allocator>::extract_long_double() const
{

if (!(isfinite)())
{
if ((isnan)())
{
return std::numeric_limits<long double>::quiet_NaN();
}
else
{
return ((!neg) ? std::numeric_limits<long double>::infinity()
: -std::numeric_limits<long double>::infinity());
}
}

cpp_dec_float<Digits10, ExponentType, Allocator> xx(*this);
if (xx.isneg())
xx.negate();

if (iszero() || (xx.compare(long_double_min()) < 0))
{
return static_cast<long double>(0.0);
}

if (xx.compare(long_double_max()) > 0)
{
return ((!neg) ? std::numeric_limits<long double>::infinity()
: -std::numeric_limits<long double>::infinity());
}

std::stringstream ss;
ss.imbue(std::locale::classic());

ss << str(std::numeric_limits<long double>::digits10 + (2 + 1), std::ios_base::scientific);

long double ld;
ss >> ld;

return ld;
}

template <unsigned Digits10, class ExponentType, class Allocator>
boost::long_long_type cpp_dec_float<Digits10, ExponentType, Allocator>::extract_signed_long_long() const
{

if (exp < static_cast<ExponentType>(0))
{
return static_cast<boost::long_long_type>(0);
}

const bool b_neg = isneg();

boost::ulong_long_type val;

if ((!b_neg) && (compare(long_long_max()) > 0))
{
return (std::numeric_limits<boost::long_long_type>::max)();
}
else if (b_neg && (compare(long_long_min()) < 0))
{
return (std::numeric_limits<boost::long_long_type>::min)();
}
else
{
cpp_dec_float<Digits10, ExponentType, Allocator> xn(extract_integer_part());
if (xn.isneg())
xn.negate();

val = static_cast<boost::ulong_long_type>(xn.data[0]);

const boost::int32_t imax = (std::min)(static_cast<boost::int32_t>(static_cast<boost::int32_t>(xn.exp) / cpp_dec_float_elem_digits10), static_cast<boost::int32_t>(cpp_dec_float_elem_number - static_cast<boost::int32_t>(1)));

for (boost::int32_t i = static_cast<boost::int32_t>(1); i <= imax; i++)
{
val *= static_cast<boost::ulong_long_type>(cpp_dec_float_elem_mask);
val += static_cast<boost::ulong_long_type>(xn.data[i]);
}
}

if (!b_neg)
{
return static_cast<boost::long_long_type>(val);
}
else
{
boost::long_long_type sval = static_cast<boost::long_long_type>(val - 1);
sval                       = -sval;
--sval;
return sval;
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
boost::ulong_long_type cpp_dec_float<Digits10, ExponentType, Allocator>::extract_unsigned_long_long() const
{

if (isneg())
{
return static_cast<boost::ulong_long_type>(extract_signed_long_long());
}

if (exp < static_cast<ExponentType>(0))
{
return static_cast<boost::ulong_long_type>(0u);
}

const cpp_dec_float<Digits10, ExponentType, Allocator> xn(extract_integer_part());

boost::ulong_long_type val;

if (xn.compare(ulong_long_max()) > 0)
{
return (std::numeric_limits<boost::ulong_long_type>::max)();
}
else
{
val = static_cast<boost::ulong_long_type>(xn.data[0]);

const boost::int32_t imax = (std::min)(static_cast<boost::int32_t>(static_cast<boost::int32_t>(xn.exp) / cpp_dec_float_elem_digits10), static_cast<boost::int32_t>(cpp_dec_float_elem_number - static_cast<boost::int32_t>(1)));

for (boost::int32_t i = static_cast<boost::int32_t>(1); i <= imax; i++)
{
val *= static_cast<boost::ulong_long_type>(cpp_dec_float_elem_mask);
val += static_cast<boost::ulong_long_type>(xn.data[i]);
}
}

return val;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator> cpp_dec_float<Digits10, ExponentType, Allocator>::extract_integer_part() const
{

if (!(isfinite)())
{
return *this;
}

if (exp < static_cast<ExponentType>(0))
{
return zero();
}


cpp_dec_float<Digits10, ExponentType, Allocator> x = *this;

const size_t first_clear = (static_cast<size_t>(x.exp) / static_cast<size_t>(cpp_dec_float_elem_digits10)) + 1u;
const size_t last_clear  = static_cast<size_t>(cpp_dec_float_elem_number);

if (first_clear < last_clear)
std::fill(x.data.begin() + first_clear, x.data.begin() + last_clear, static_cast<boost::uint32_t>(0u));

return x;
}

template <unsigned Digits10, class ExponentType, class Allocator>
std::string cpp_dec_float<Digits10, ExponentType, Allocator>::str(boost::intmax_t number_of_digits, std::ios_base::fmtflags f) const
{
if ((this->isinf)())
{
if (this->isneg())
return "-inf";
else if (f & std::ios_base::showpos)
return "+inf";
else
return "inf";
}
else if ((this->isnan)())
{
return "nan";
}

std::string     str;
boost::intmax_t org_digits(number_of_digits);
ExponentType    my_exp = order();

if (number_of_digits == 0)
number_of_digits = cpp_dec_float_total_digits10;

if (f & std::ios_base::fixed)
{
number_of_digits += my_exp + 1;
}
else if (f & std::ios_base::scientific)
++number_of_digits;
const std::size_t number_of_elements = (std::min)(static_cast<std::size_t>((number_of_digits / static_cast<std::size_t>(cpp_dec_float_elem_digits10)) + 2u),
static_cast<std::size_t>(cpp_dec_float_elem_number));

std::stringstream ss;
ss.imbue(std::locale::classic());
ss << data[0];
for (std::size_t i = static_cast<std::size_t>(1u); i < number_of_elements; i++)
{
ss << std::setw(static_cast<std::streamsize>(cpp_dec_float_elem_digits10))
<< std::setfill(static_cast<char>('0'))
<< data[i];
}
str += ss.str();

bool have_leading_zeros = false;

if (number_of_digits == 0)
{
number_of_digits -= my_exp + 1; 
str.insert(static_cast<std::string::size_type>(0), std::string::size_type(number_of_digits), '0');
have_leading_zeros = true;
}

if (number_of_digits < 0)
{
str = "0";
if (isneg())
str.insert(static_cast<std::string::size_type>(0), 1, '-');
boost::multiprecision::detail::format_float_string(str, 0, number_of_digits - my_exp - 1, f, this->iszero());
return str;
}
else
{
if (str.length() > static_cast<std::string::size_type>(number_of_digits))
{
const boost::uint32_t round = static_cast<boost::uint32_t>(static_cast<boost::uint32_t>(str[static_cast<std::string::size_type>(number_of_digits)]) - static_cast<boost::uint32_t>('0'));

bool need_round_up = round >= 5u;

if (round == 5u)
{
const boost::uint32_t ix = static_cast<boost::uint32_t>(static_cast<boost::uint32_t>(str[static_cast<std::string::size_type>(number_of_digits - 1)]) - static_cast<boost::uint32_t>('0'));
if ((ix & 1u) == 0)
{
if (str.find_first_not_of('0', static_cast<std::string::size_type>(number_of_digits + 1)) == std::string::npos)
{
bool all_zeros = true;
for (std::size_t i = number_of_elements; i < data.size(); i++)
{
if (data[i])
{
all_zeros = false;
break;
}
}
if (all_zeros)
need_round_up = false; 
}
}
}

str.erase(static_cast<std::string::size_type>(number_of_digits));

if (need_round_up)
{
std::size_t ix = static_cast<std::size_t>(str.length() - 1u);

while (ix && (static_cast<boost::int32_t>(str.at(ix)) - static_cast<boost::int32_t>('0') == static_cast<boost::int32_t>(9)))
{
str.at(ix) = static_cast<char>('0');
--ix;
}

if (!ix)
{
if (static_cast<boost::int32_t>(static_cast<boost::int32_t>(str.at(ix)) - static_cast<boost::int32_t>(0x30)) == static_cast<boost::int32_t>(9))
{
str.at(ix) = static_cast<char>('1');
++my_exp;
}
else
{
++str.at(ix);
}
}
else
{
++str[ix];
}
}
}
}

if (have_leading_zeros)
{
if (str[std::string::size_type(number_of_digits - 1)] != '0')
{
++my_exp;
str.erase(0, std::string::size_type(number_of_digits - 1));
}
else
str.erase(0, std::string::size_type(number_of_digits));
}

if (isneg())
str.insert(static_cast<std::string::size_type>(0), 1, '-');

boost::multiprecision::detail::format_float_string(str, my_exp, org_digits, f, this->iszero());
return str;
}

template <unsigned Digits10, class ExponentType, class Allocator>
bool cpp_dec_float<Digits10, ExponentType, Allocator>::rd_string(const char* const s)
{
#ifndef BOOST_NO_EXCEPTIONS
try
{
#endif

std::string str(s);


exp = static_cast<ExponentType>(0);

std::size_t pos;

if (((pos = str.find('e')) != std::string::npos) || ((pos = str.find('E')) != std::string::npos))
{
exp = boost::lexical_cast<ExponentType>(static_cast<const char*>(str.c_str() + (pos + 1u)));
str = str.substr(static_cast<std::size_t>(0u), pos);
}

neg = false;

if (str.size())
{
if (str[0] == '-')
{
neg = true;
str.erase(0, 1);
}
else if (str[0] == '+')
{
str.erase(0, 1);
}
}
if ((str == "inf") || (str == "INF") || (str == "infinity") || (str == "INFINITY"))
{
if (neg)
{
*this = this->inf();
this->negate();
}
else
*this = this->inf();
return true;
}
if ((str.size() >= 3) && ((str.substr(0, 3) == "nan") || (str.substr(0, 3) == "NAN") || (str.substr(0, 3) == "NaN")))
{
*this = this->nan();
return true;
}

const std::string::iterator fwd_it_leading_zero = std::find_if(str.begin(), str.end(), char_is_nonzero_predicate);

if (fwd_it_leading_zero != str.begin())
{
if (fwd_it_leading_zero == str.end())
{
operator=(zero());
return true;
}
else
{
str.erase(str.begin(), fwd_it_leading_zero);
}
}


pos = str.find(static_cast<char>('.'));

if (pos != std::string::npos)
{
const std::string::const_reverse_iterator rit_non_zero = std::find_if(str.rbegin(), str.rend(), char_is_nonzero_predicate);

if (rit_non_zero != static_cast<std::string::const_reverse_iterator>(str.rbegin()))
{
const std::string::size_type ofs = str.length() - std::distance<std::string::const_reverse_iterator>(str.rbegin(), rit_non_zero);
str.erase(str.begin() + ofs, str.end());
}

if (str == std::string("."))
{
operator=(zero());
return true;
}

if (str.at(static_cast<std::size_t>(0u)) == static_cast<char>('.'))
{
const std::string::iterator it_non_zero = std::find_if(str.begin() + 1u, str.end(), char_is_nonzero_predicate);

std::size_t delta_exp = static_cast<std::size_t>(0u);

if (str.at(static_cast<std::size_t>(1u)) == static_cast<char>('0'))
{
delta_exp = std::distance<std::string::const_iterator>(str.begin() + 1u, it_non_zero);
}

str.erase(str.begin(), it_non_zero);
str.insert(static_cast<std::string::size_type>(1u), ".");
exp -= static_cast<ExponentType>(delta_exp + 1u);
}
}
else
{
str.append(".");
}

std::size_t       n_shift   = static_cast<std::size_t>(0u);
const std::size_t n_exp_rem = static_cast<std::size_t>(exp % static_cast<ExponentType>(cpp_dec_float_elem_digits10));

if ((exp % static_cast<ExponentType>(cpp_dec_float_elem_digits10)) != static_cast<ExponentType>(0))
{
n_shift = ((exp < static_cast<ExponentType>(0))
? static_cast<std::size_t>(n_exp_rem + static_cast<std::size_t>(cpp_dec_float_elem_digits10))
: static_cast<std::size_t>(n_exp_rem));
}

pos = str.find(static_cast<char>('.'));

std::size_t pos_plus_one = static_cast<std::size_t>(pos + 1u);

if ((str.length() - pos_plus_one) < n_shift)
{
const std::size_t sz = static_cast<std::size_t>(n_shift - (str.length() - pos_plus_one));

str.append(std::string(sz, static_cast<char>('0')));
}

if (n_shift != static_cast<std::size_t>(0u))
{
str.insert(static_cast<std::string::size_type>(pos_plus_one + n_shift), ".");

str.erase(pos, static_cast<std::string::size_type>(1u));

exp -= static_cast<ExponentType>(n_shift);
}

pos          = str.find(static_cast<char>('.'));
pos_plus_one = static_cast<std::size_t>(pos + 1u);

if (pos > static_cast<std::size_t>(cpp_dec_float_elem_digits10))
{
const boost::int32_t n_pos         = static_cast<boost::int32_t>(pos);
const boost::int32_t n_rem_is_zero = ((static_cast<boost::int32_t>(n_pos % cpp_dec_float_elem_digits10) == static_cast<boost::int32_t>(0)) ? static_cast<boost::int32_t>(1) : static_cast<boost::int32_t>(0));
const boost::int32_t n             = static_cast<boost::int32_t>(static_cast<boost::int32_t>(n_pos / cpp_dec_float_elem_digits10) - n_rem_is_zero);

str.insert(static_cast<std::size_t>(static_cast<boost::int32_t>(n_pos - static_cast<boost::int32_t>(n * cpp_dec_float_elem_digits10))), ".");

str.erase(pos_plus_one, static_cast<std::size_t>(1u));

exp += static_cast<ExponentType>(static_cast<ExponentType>(n) * static_cast<ExponentType>(cpp_dec_float_elem_digits10));
}

pos          = str.find(static_cast<char>('.'));
pos_plus_one = static_cast<std::size_t>(pos + 1u);

const boost::int32_t n_dec = static_cast<boost::int32_t>(static_cast<boost::int32_t>(str.length() - 1u) - static_cast<boost::int32_t>(pos));
const boost::int32_t n_rem = static_cast<boost::int32_t>(n_dec % cpp_dec_float_elem_digits10);

boost::int32_t n_cnt = ((n_rem != static_cast<boost::int32_t>(0))
? static_cast<boost::int32_t>(cpp_dec_float_elem_digits10 - n_rem)
: static_cast<boost::int32_t>(0));

if (n_cnt != static_cast<boost::int32_t>(0))
{
str.append(static_cast<std::size_t>(n_cnt), static_cast<char>('0'));
}

const std::size_t max_dec = static_cast<std::size_t>((cpp_dec_float_elem_number - 1) * cpp_dec_float_elem_digits10);

if (static_cast<std::size_t>(str.length() - pos) > max_dec)
{
str = str.substr(static_cast<std::size_t>(0u),
static_cast<std::size_t>(pos_plus_one + max_dec));
}


std::fill(data.begin(), data.end(), static_cast<boost::uint32_t>(0u));


data[0u] = boost::lexical_cast<boost::uint32_t>(str.substr(static_cast<std::size_t>(0u), pos));

const std::string::size_type i_end = ((str.length() - pos_plus_one) / static_cast<std::string::size_type>(cpp_dec_float_elem_digits10));

for (std::string::size_type i = static_cast<std::string::size_type>(0u); i < i_end; i++)
{
const std::string::const_iterator it = str.begin() + pos_plus_one + (i * static_cast<std::string::size_type>(cpp_dec_float_elem_digits10));

data[i + 1u] = boost::lexical_cast<boost::uint32_t>(std::string(it, it + static_cast<std::string::size_type>(cpp_dec_float_elem_digits10)));
}

if (exp > cpp_dec_float_max_exp10)
{
const bool b_result_is_neg = neg;

*this = inf();
if (b_result_is_neg)
negate();
}

if (exp <= cpp_dec_float_min_exp10)
{
if (exp == cpp_dec_float_min_exp10)
{
cpp_dec_float<Digits10, ExponentType, Allocator> test = *this;

test.exp = static_cast<ExponentType>(0);

if (test.isone())
{
*this = zero();
}
}
else
{
*this = zero();
}
}

#ifndef BOOST_NO_EXCEPTIONS
}
catch (const bad_lexical_cast&)
{
std::string msg = "Unable to parse the string \"";
msg += s;
msg += "\" as a floating point value.";
throw std::runtime_error(msg);
}
#endif
return true;
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float(const double mantissa, const ExponentType exponent)
: data(),
exp(static_cast<ExponentType>(0)),
neg(false),
fpclass(cpp_dec_float_finite),
prec_elem(cpp_dec_float_elem_number)
{

const bool mantissa_is_iszero = (::fabs(mantissa) < ((std::numeric_limits<double>::min)() * (1.0 + std::numeric_limits<double>::epsilon())));

if (mantissa_is_iszero)
{
std::fill(data.begin(), data.end(), static_cast<boost::uint32_t>(0u));
return;
}

const bool b_neg = (mantissa < 0.0);

double       d = ((!b_neg) ? mantissa : -mantissa);
ExponentType e = exponent;

while (d > 10.0)
{
d /= 10.0;
++e;
}
while (d < 1.0)
{
d *= 10.0;
--e;
}

boost::int32_t shift = static_cast<boost::int32_t>(e % static_cast<boost::int32_t>(cpp_dec_float_elem_digits10));

while (static_cast<boost::int32_t>(shift-- % cpp_dec_float_elem_digits10) != static_cast<boost::int32_t>(0))
{
d *= 10.0;
--e;
}

exp = e;
neg = b_neg;

std::fill(data.begin(), data.end(), static_cast<boost::uint32_t>(0u));

static const boost::int32_t digit_ratio = static_cast<boost::int32_t>(static_cast<boost::int32_t>(std::numeric_limits<double>::digits10) / static_cast<boost::int32_t>(cpp_dec_float_elem_digits10));
static const boost::int32_t digit_loops = static_cast<boost::int32_t>(digit_ratio + static_cast<boost::int32_t>(2));

for (boost::int32_t i = static_cast<boost::int32_t>(0); i < digit_loops; i++)
{
boost::uint32_t n = static_cast<boost::uint32_t>(static_cast<boost::uint64_t>(d));
data[i]           = static_cast<boost::uint32_t>(n);
d -= static_cast<double>(n);
d *= static_cast<double>(cpp_dec_float_elem_mask);
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
template <class Float>
typename boost::enable_if_c<boost::is_floating_point<Float>::value, cpp_dec_float<Digits10, ExponentType, Allocator>&>::type cpp_dec_float<Digits10, ExponentType, Allocator>::operator=(Float a)
{
using std::floor;
using std::frexp;
using std::ldexp;

if (a == 0)
return *this = zero();

if (a == 1)
return *this = one();

if ((boost::math::isinf)(a))
{
*this = inf();
if (a < 0)
this->negate();
return *this;
}

if ((boost::math::isnan)(a))
return *this = nan();

int         e;
Float f, term;
*this = zero();

f = frexp(a, &e);
BOOST_ASSERT((boost::math::isfinite)(f));

static const int shift = std::numeric_limits<int>::digits - 1;

while (f)
{
f = ldexp(f, shift);
BOOST_ASSERT((boost::math::isfinite)(f));
term = floor(f);
e -= shift;
*this *= pow2(shift);
if (term > 0)
add_unsigned_long_long(static_cast<unsigned>(term));
else
sub_unsigned_long_long(static_cast<unsigned>(-term));
f -= term;
}

if (e != 0)
*this *= pow2(e);

return *this;
}

template <unsigned Digits10, class ExponentType, class Allocator>
void cpp_dec_float<Digits10, ExponentType, Allocator>::from_unsigned_long_long(const boost::ulong_long_type u)
{
std::fill(data.begin(), data.end(), static_cast<boost::uint32_t>(0u));

exp       = static_cast<ExponentType>(0);
neg       = false;
fpclass   = cpp_dec_float_finite;
prec_elem = cpp_dec_float_elem_number;

if (u == 0)
{
return;
}

std::size_t i = static_cast<std::size_t>(0u);

boost::ulong_long_type uu = u;

boost::uint32_t temp[(std::numeric_limits<boost::ulong_long_type>::digits10 / static_cast<int>(cpp_dec_float_elem_digits10)) + 3] = {static_cast<boost::uint32_t>(0u)};

while (uu != static_cast<boost::ulong_long_type>(0u))
{
temp[i] = static_cast<boost::uint32_t>(uu % static_cast<boost::ulong_long_type>(cpp_dec_float_elem_mask));
uu      = static_cast<boost::ulong_long_type>(uu / static_cast<boost::ulong_long_type>(cpp_dec_float_elem_mask));
++i;
}

if (i > static_cast<std::size_t>(1u))
{
exp += static_cast<ExponentType>((i - 1u) * static_cast<std::size_t>(cpp_dec_float_elem_digits10));
}

std::reverse(temp, temp + i);
std::copy(temp, temp + (std::min)(i, static_cast<std::size_t>(cpp_dec_float_elem_number)), data.begin());
}

template <unsigned Digits10, class ExponentType, class Allocator>
boost::uint32_t cpp_dec_float<Digits10, ExponentType, Allocator>::mul_loop_uv(boost::uint32_t* const u, const boost::uint32_t* const v, const boost::int32_t p)
{
BOOST_STATIC_ASSERT_MSG(cpp_dec_float_elem_number < 1800, "Too many limbs in the data type for the multiplication algorithm - unsupported precision in cpp_dec_float.");

boost::uint64_t carry = static_cast<boost::uint64_t>(0u);

for (boost::int32_t j = static_cast<boost::int32_t>(p - 1u); j >= static_cast<boost::int32_t>(0); j--)
{
boost::uint64_t sum = carry;

for (boost::int32_t i = j; i >= static_cast<boost::int32_t>(0); i--)
{
sum += static_cast<boost::uint64_t>(u[j - i] * static_cast<boost::uint64_t>(v[i]));
}

u[j]  = static_cast<boost::uint32_t>(sum % static_cast<boost::uint32_t>(cpp_dec_float_elem_mask));
carry = static_cast<boost::uint64_t>(sum / static_cast<boost::uint32_t>(cpp_dec_float_elem_mask));
}

return static_cast<boost::uint32_t>(carry);
}

template <unsigned Digits10, class ExponentType, class Allocator>
boost::uint32_t cpp_dec_float<Digits10, ExponentType, Allocator>::mul_loop_n(boost::uint32_t* const u, boost::uint32_t n, const boost::int32_t p)
{
boost::uint64_t carry = static_cast<boost::uint64_t>(0u);

for (boost::int32_t j = p - 1; j >= static_cast<boost::int32_t>(0); j--)
{
const boost::uint64_t t = static_cast<boost::uint64_t>(carry + static_cast<boost::uint64_t>(u[j] * static_cast<boost::uint64_t>(n)));
carry                   = static_cast<boost::uint64_t>(t / static_cast<boost::uint32_t>(cpp_dec_float_elem_mask));
u[j]                    = static_cast<boost::uint32_t>(t - static_cast<boost::uint64_t>(static_cast<boost::uint32_t>(cpp_dec_float_elem_mask) * static_cast<boost::uint64_t>(carry)));
}

return static_cast<boost::uint32_t>(carry);
}

template <unsigned Digits10, class ExponentType, class Allocator>
boost::uint32_t cpp_dec_float<Digits10, ExponentType, Allocator>::div_loop_n(boost::uint32_t* const u, boost::uint32_t n, const boost::int32_t p)
{
boost::uint64_t prev = static_cast<boost::uint64_t>(0u);

for (boost::int32_t j = static_cast<boost::int32_t>(0); j < p; j++)
{
const boost::uint64_t t = static_cast<boost::uint64_t>(u[j] + static_cast<boost::uint64_t>(prev * static_cast<boost::uint32_t>(cpp_dec_float_elem_mask)));
u[j]                    = static_cast<boost::uint32_t>(t / n);
prev                    = static_cast<boost::uint64_t>(t - static_cast<boost::uint64_t>(n * static_cast<boost::uint64_t>(u[j])));
}

return static_cast<boost::uint32_t>(prev);
}

template <unsigned Digits10, class ExponentType, class Allocator>
cpp_dec_float<Digits10, ExponentType, Allocator> cpp_dec_float<Digits10, ExponentType, Allocator>::pow2(const boost::long_long_type p)
{
init.do_nothing();
static const boost::array<cpp_dec_float<Digits10, ExponentType, Allocator>, 255u> p2_data =
{{cpp_dec_float("5.877471754111437539843682686111228389093327783860437607543758531392086297273635864257812500000000000e-39"),
cpp_dec_float("1.175494350822287507968736537222245677818665556772087521508751706278417259454727172851562500000000000e-38"),
cpp_dec_float("2.350988701644575015937473074444491355637331113544175043017503412556834518909454345703125000000000000e-38"),
cpp_dec_float("4.701977403289150031874946148888982711274662227088350086035006825113669037818908691406250000000000000e-38"),
cpp_dec_float("9.403954806578300063749892297777965422549324454176700172070013650227338075637817382812500000000000000e-38"),
cpp_dec_float("1.880790961315660012749978459555593084509864890835340034414002730045467615127563476562500000000000000e-37"),
cpp_dec_float("3.761581922631320025499956919111186169019729781670680068828005460090935230255126953125000000000000000e-37"),
cpp_dec_float("7.523163845262640050999913838222372338039459563341360137656010920181870460510253906250000000000000000e-37"),
cpp_dec_float("1.504632769052528010199982767644474467607891912668272027531202184036374092102050781250000000000000000e-36"),
cpp_dec_float("3.009265538105056020399965535288948935215783825336544055062404368072748184204101562500000000000000000e-36"),
cpp_dec_float("6.018531076210112040799931070577897870431567650673088110124808736145496368408203125000000000000000000e-36"),
cpp_dec_float("1.203706215242022408159986214115579574086313530134617622024961747229099273681640625000000000000000000e-35"),
cpp_dec_float("2.407412430484044816319972428231159148172627060269235244049923494458198547363281250000000000000000000e-35"),
cpp_dec_float("4.814824860968089632639944856462318296345254120538470488099846988916397094726562500000000000000000000e-35"),
cpp_dec_float("9.629649721936179265279889712924636592690508241076940976199693977832794189453125000000000000000000000e-35"),
cpp_dec_float("1.925929944387235853055977942584927318538101648215388195239938795566558837890625000000000000000000000e-34"),
cpp_dec_float("3.851859888774471706111955885169854637076203296430776390479877591133117675781250000000000000000000000e-34"),
cpp_dec_float("7.703719777548943412223911770339709274152406592861552780959755182266235351562500000000000000000000000e-34"),
cpp_dec_float("1.540743955509788682444782354067941854830481318572310556191951036453247070312500000000000000000000000e-33"),
cpp_dec_float("3.081487911019577364889564708135883709660962637144621112383902072906494140625000000000000000000000000e-33"),
cpp_dec_float("6.162975822039154729779129416271767419321925274289242224767804145812988281250000000000000000000000000e-33"),
cpp_dec_float("1.232595164407830945955825883254353483864385054857848444953560829162597656250000000000000000000000000e-32"),
cpp_dec_float("2.465190328815661891911651766508706967728770109715696889907121658325195312500000000000000000000000000e-32"),
cpp_dec_float("4.930380657631323783823303533017413935457540219431393779814243316650390625000000000000000000000000000e-32"),
cpp_dec_float("9.860761315262647567646607066034827870915080438862787559628486633300781250000000000000000000000000000e-32"),
cpp_dec_float("1.972152263052529513529321413206965574183016087772557511925697326660156250000000000000000000000000000e-31"),
cpp_dec_float("3.944304526105059027058642826413931148366032175545115023851394653320312500000000000000000000000000000e-31"),
cpp_dec_float("7.888609052210118054117285652827862296732064351090230047702789306640625000000000000000000000000000000e-31"),
cpp_dec_float("1.577721810442023610823457130565572459346412870218046009540557861328125000000000000000000000000000000e-30"),
cpp_dec_float("3.155443620884047221646914261131144918692825740436092019081115722656250000000000000000000000000000000e-30"),
cpp_dec_float("6.310887241768094443293828522262289837385651480872184038162231445312500000000000000000000000000000000e-30"),
cpp_dec_float("1.262177448353618888658765704452457967477130296174436807632446289062500000000000000000000000000000000e-29"),
cpp_dec_float("2.524354896707237777317531408904915934954260592348873615264892578125000000000000000000000000000000000e-29"),
cpp_dec_float("5.048709793414475554635062817809831869908521184697747230529785156250000000000000000000000000000000000e-29"),
cpp_dec_float("1.009741958682895110927012563561966373981704236939549446105957031250000000000000000000000000000000000e-28"),
cpp_dec_float("2.019483917365790221854025127123932747963408473879098892211914062500000000000000000000000000000000000e-28"),
cpp_dec_float("4.038967834731580443708050254247865495926816947758197784423828125000000000000000000000000000000000000e-28"),
cpp_dec_float("8.077935669463160887416100508495730991853633895516395568847656250000000000000000000000000000000000000e-28"),
cpp_dec_float("1.615587133892632177483220101699146198370726779103279113769531250000000000000000000000000000000000000e-27"),
cpp_dec_float("3.231174267785264354966440203398292396741453558206558227539062500000000000000000000000000000000000000e-27"),
cpp_dec_float("6.462348535570528709932880406796584793482907116413116455078125000000000000000000000000000000000000000e-27"),
cpp_dec_float("1.292469707114105741986576081359316958696581423282623291015625000000000000000000000000000000000000000e-26"),
cpp_dec_float("2.584939414228211483973152162718633917393162846565246582031250000000000000000000000000000000000000000e-26"),
cpp_dec_float("5.169878828456422967946304325437267834786325693130493164062500000000000000000000000000000000000000000e-26"),
cpp_dec_float("1.033975765691284593589260865087453566957265138626098632812500000000000000000000000000000000000000000e-25"),
cpp_dec_float("2.067951531382569187178521730174907133914530277252197265625000000000000000000000000000000000000000000e-25"),
cpp_dec_float("4.135903062765138374357043460349814267829060554504394531250000000000000000000000000000000000000000000e-25"),
cpp_dec_float("8.271806125530276748714086920699628535658121109008789062500000000000000000000000000000000000000000000e-25"),
cpp_dec_float("1.654361225106055349742817384139925707131624221801757812500000000000000000000000000000000000000000000e-24"),
cpp_dec_float("3.308722450212110699485634768279851414263248443603515625000000000000000000000000000000000000000000000e-24"),
cpp_dec_float("6.617444900424221398971269536559702828526496887207031250000000000000000000000000000000000000000000000e-24"),
cpp_dec_float("1.323488980084844279794253907311940565705299377441406250000000000000000000000000000000000000000000000e-23"),
cpp_dec_float("2.646977960169688559588507814623881131410598754882812500000000000000000000000000000000000000000000000e-23"),
cpp_dec_float("5.293955920339377119177015629247762262821197509765625000000000000000000000000000000000000000000000000e-23"),
cpp_dec_float("1.058791184067875423835403125849552452564239501953125000000000000000000000000000000000000000000000000e-22"),
cpp_dec_float("2.117582368135750847670806251699104905128479003906250000000000000000000000000000000000000000000000000e-22"),
cpp_dec_float("4.235164736271501695341612503398209810256958007812500000000000000000000000000000000000000000000000000e-22"),
cpp_dec_float("8.470329472543003390683225006796419620513916015625000000000000000000000000000000000000000000000000000e-22"),
cpp_dec_float("1.694065894508600678136645001359283924102783203125000000000000000000000000000000000000000000000000000e-21"),
cpp_dec_float("3.388131789017201356273290002718567848205566406250000000000000000000000000000000000000000000000000000e-21"),
cpp_dec_float("6.776263578034402712546580005437135696411132812500000000000000000000000000000000000000000000000000000e-21"),
cpp_dec_float("1.355252715606880542509316001087427139282226562500000000000000000000000000000000000000000000000000000e-20"),
cpp_dec_float("2.710505431213761085018632002174854278564453125000000000000000000000000000000000000000000000000000000e-20"),
cpp_dec_float("5.421010862427522170037264004349708557128906250000000000000000000000000000000000000000000000000000000e-20"),
cpp_dec_float("1.084202172485504434007452800869941711425781250000000000000000000000000000000000000000000000000000000e-19"),
cpp_dec_float("2.168404344971008868014905601739883422851562500000000000000000000000000000000000000000000000000000000e-19"),
cpp_dec_float("4.336808689942017736029811203479766845703125000000000000000000000000000000000000000000000000000000000e-19"),
cpp_dec_float("8.673617379884035472059622406959533691406250000000000000000000000000000000000000000000000000000000000e-19"),
cpp_dec_float("1.734723475976807094411924481391906738281250000000000000000000000000000000000000000000000000000000000e-18"),
cpp_dec_float("3.469446951953614188823848962783813476562500000000000000000000000000000000000000000000000000000000000e-18"),
cpp_dec_float("6.938893903907228377647697925567626953125000000000000000000000000000000000000000000000000000000000000e-18"),
cpp_dec_float("1.387778780781445675529539585113525390625000000000000000000000000000000000000000000000000000000000000e-17"),
cpp_dec_float("2.775557561562891351059079170227050781250000000000000000000000000000000000000000000000000000000000000e-17"),
cpp_dec_float("5.551115123125782702118158340454101562500000000000000000000000000000000000000000000000000000000000000e-17"),
cpp_dec_float("1.110223024625156540423631668090820312500000000000000000000000000000000000000000000000000000000000000e-16"),
cpp_dec_float("2.220446049250313080847263336181640625000000000000000000000000000000000000000000000000000000000000000e-16"),
cpp_dec_float("4.440892098500626161694526672363281250000000000000000000000000000000000000000000000000000000000000000e-16"),
cpp_dec_float("8.881784197001252323389053344726562500000000000000000000000000000000000000000000000000000000000000000e-16"),
cpp_dec_float("1.776356839400250464677810668945312500000000000000000000000000000000000000000000000000000000000000000e-15"),
cpp_dec_float("3.552713678800500929355621337890625000000000000000000000000000000000000000000000000000000000000000000e-15"),
cpp_dec_float("7.105427357601001858711242675781250000000000000000000000000000000000000000000000000000000000000000000e-15"),
cpp_dec_float("1.421085471520200371742248535156250000000000000000000000000000000000000000000000000000000000000000000e-14"),
cpp_dec_float("2.842170943040400743484497070312500000000000000000000000000000000000000000000000000000000000000000000e-14"),
cpp_dec_float("5.684341886080801486968994140625000000000000000000000000000000000000000000000000000000000000000000000e-14"),
cpp_dec_float("1.136868377216160297393798828125000000000000000000000000000000000000000000000000000000000000000000000e-13"),
cpp_dec_float("2.273736754432320594787597656250000000000000000000000000000000000000000000000000000000000000000000000e-13"),
cpp_dec_float("4.547473508864641189575195312500000000000000000000000000000000000000000000000000000000000000000000000e-13"),
cpp_dec_float("9.094947017729282379150390625000000000000000000000000000000000000000000000000000000000000000000000000e-13"),
cpp_dec_float("1.818989403545856475830078125000000000000000000000000000000000000000000000000000000000000000000000000e-12"),
cpp_dec_float("3.637978807091712951660156250000000000000000000000000000000000000000000000000000000000000000000000000e-12"),
cpp_dec_float("7.275957614183425903320312500000000000000000000000000000000000000000000000000000000000000000000000000e-12"),
cpp_dec_float("1.455191522836685180664062500000000000000000000000000000000000000000000000000000000000000000000000000e-11"),
cpp_dec_float("2.910383045673370361328125000000000000000000000000000000000000000000000000000000000000000000000000000e-11"),
cpp_dec_float("5.820766091346740722656250000000000000000000000000000000000000000000000000000000000000000000000000000e-11"),
cpp_dec_float("1.164153218269348144531250000000000000000000000000000000000000000000000000000000000000000000000000000e-10"),
cpp_dec_float("2.328306436538696289062500000000000000000000000000000000000000000000000000000000000000000000000000000e-10"),
cpp_dec_float("4.656612873077392578125000000000000000000000000000000000000000000000000000000000000000000000000000000e-10"),
cpp_dec_float("9.313225746154785156250000000000000000000000000000000000000000000000000000000000000000000000000000000e-10"),
cpp_dec_float("1.862645149230957031250000000000000000000000000000000000000000000000000000000000000000000000000000000e-9"),
cpp_dec_float("3.725290298461914062500000000000000000000000000000000000000000000000000000000000000000000000000000000e-9"),
cpp_dec_float("7.450580596923828125000000000000000000000000000000000000000000000000000000000000000000000000000000000e-9"),
cpp_dec_float("1.490116119384765625000000000000000000000000000000000000000000000000000000000000000000000000000000000e-8"),
cpp_dec_float("2.980232238769531250000000000000000000000000000000000000000000000000000000000000000000000000000000000e-8"),
cpp_dec_float("5.960464477539062500000000000000000000000000000000000000000000000000000000000000000000000000000000000e-8"),
cpp_dec_float("1.192092895507812500000000000000000000000000000000000000000000000000000000000000000000000000000000000e-7"),
cpp_dec_float("2.384185791015625000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-7"),
cpp_dec_float("4.768371582031250000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-7"),
cpp_dec_float("9.536743164062500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-7"),
cpp_dec_float("1.907348632812500000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-6"),
cpp_dec_float("3.814697265625000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-6"),
cpp_dec_float("7.629394531250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e-6"),
cpp_dec_float("0.000015258789062500000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.000030517578125000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.000061035156250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.000122070312500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.000244140625000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.000488281250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.000976562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.001953125000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.003906250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.007812500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.01562500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.03125000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.06250000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
cpp_dec_float("0.125"),
cpp_dec_float("0.25"),
cpp_dec_float("0.5"),
one(),
two(),
cpp_dec_float(static_cast<boost::ulong_long_type>(4)),
cpp_dec_float(static_cast<boost::ulong_long_type>(8)),
cpp_dec_float(static_cast<boost::ulong_long_type>(16)),
cpp_dec_float(static_cast<boost::ulong_long_type>(32)),
cpp_dec_float(static_cast<boost::ulong_long_type>(64)),
cpp_dec_float(static_cast<boost::ulong_long_type>(128)),
cpp_dec_float(static_cast<boost::ulong_long_type>(256)),
cpp_dec_float(static_cast<boost::ulong_long_type>(512)),
cpp_dec_float(static_cast<boost::ulong_long_type>(1024)),
cpp_dec_float(static_cast<boost::ulong_long_type>(2048)),
cpp_dec_float(static_cast<boost::ulong_long_type>(4096)),
cpp_dec_float(static_cast<boost::ulong_long_type>(8192)),
cpp_dec_float(static_cast<boost::ulong_long_type>(16384)),
cpp_dec_float(static_cast<boost::ulong_long_type>(32768)),
cpp_dec_float(static_cast<boost::ulong_long_type>(65536)),
cpp_dec_float(static_cast<boost::ulong_long_type>(131072)),
cpp_dec_float(static_cast<boost::ulong_long_type>(262144)),
cpp_dec_float(static_cast<boost::ulong_long_type>(524288)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 20u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 21u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 22u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 23u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 24u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 25u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 26u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 27u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 28u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 29u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 30u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uL << 31u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 32u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 33u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 34u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 35u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 36u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 37u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 38u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 39u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 40u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 41u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 42u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 43u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 44u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 45u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 46u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 47u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 48u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 49u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 50u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 51u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 52u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 53u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 54u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 55u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 56u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 57u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 58u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 59u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 60u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 61u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 62u)),
cpp_dec_float(static_cast<boost::uint64_t>(1uLL << 63u)),
cpp_dec_float("1.844674407370955161600000000000000000000000000000000000000000000000000000000000000000000000000000000e19"),
cpp_dec_float("3.689348814741910323200000000000000000000000000000000000000000000000000000000000000000000000000000000e19"),
cpp_dec_float("7.378697629483820646400000000000000000000000000000000000000000000000000000000000000000000000000000000e19"),
cpp_dec_float("1.475739525896764129280000000000000000000000000000000000000000000000000000000000000000000000000000000e20"),
cpp_dec_float("2.951479051793528258560000000000000000000000000000000000000000000000000000000000000000000000000000000e20"),
cpp_dec_float("5.902958103587056517120000000000000000000000000000000000000000000000000000000000000000000000000000000e20"),
cpp_dec_float("1.180591620717411303424000000000000000000000000000000000000000000000000000000000000000000000000000000e21"),
cpp_dec_float("2.361183241434822606848000000000000000000000000000000000000000000000000000000000000000000000000000000e21"),
cpp_dec_float("4.722366482869645213696000000000000000000000000000000000000000000000000000000000000000000000000000000e21"),
cpp_dec_float("9.444732965739290427392000000000000000000000000000000000000000000000000000000000000000000000000000000e21"),
cpp_dec_float("1.888946593147858085478400000000000000000000000000000000000000000000000000000000000000000000000000000e22"),
cpp_dec_float("3.777893186295716170956800000000000000000000000000000000000000000000000000000000000000000000000000000e22"),
cpp_dec_float("7.555786372591432341913600000000000000000000000000000000000000000000000000000000000000000000000000000e22"),
cpp_dec_float("1.511157274518286468382720000000000000000000000000000000000000000000000000000000000000000000000000000e23"),
cpp_dec_float("3.022314549036572936765440000000000000000000000000000000000000000000000000000000000000000000000000000e23"),
cpp_dec_float("6.044629098073145873530880000000000000000000000000000000000000000000000000000000000000000000000000000e23"),
cpp_dec_float("1.208925819614629174706176000000000000000000000000000000000000000000000000000000000000000000000000000e24"),
cpp_dec_float("2.417851639229258349412352000000000000000000000000000000000000000000000000000000000000000000000000000e24"),
cpp_dec_float("4.835703278458516698824704000000000000000000000000000000000000000000000000000000000000000000000000000e24"),
cpp_dec_float("9.671406556917033397649408000000000000000000000000000000000000000000000000000000000000000000000000000e24"),
cpp_dec_float("1.934281311383406679529881600000000000000000000000000000000000000000000000000000000000000000000000000e25"),
cpp_dec_float("3.868562622766813359059763200000000000000000000000000000000000000000000000000000000000000000000000000e25"),
cpp_dec_float("7.737125245533626718119526400000000000000000000000000000000000000000000000000000000000000000000000000e25"),
cpp_dec_float("1.547425049106725343623905280000000000000000000000000000000000000000000000000000000000000000000000000e26"),
cpp_dec_float("3.094850098213450687247810560000000000000000000000000000000000000000000000000000000000000000000000000e26"),
cpp_dec_float("6.189700196426901374495621120000000000000000000000000000000000000000000000000000000000000000000000000e26"),
cpp_dec_float("1.237940039285380274899124224000000000000000000000000000000000000000000000000000000000000000000000000e27"),
cpp_dec_float("2.475880078570760549798248448000000000000000000000000000000000000000000000000000000000000000000000000e27"),
cpp_dec_float("4.951760157141521099596496896000000000000000000000000000000000000000000000000000000000000000000000000e27"),
cpp_dec_float("9.903520314283042199192993792000000000000000000000000000000000000000000000000000000000000000000000000e27"),
cpp_dec_float("1.980704062856608439838598758400000000000000000000000000000000000000000000000000000000000000000000000e28"),
cpp_dec_float("3.961408125713216879677197516800000000000000000000000000000000000000000000000000000000000000000000000e28"),
cpp_dec_float("7.922816251426433759354395033600000000000000000000000000000000000000000000000000000000000000000000000e28"),
cpp_dec_float("1.584563250285286751870879006720000000000000000000000000000000000000000000000000000000000000000000000e29"),
cpp_dec_float("3.169126500570573503741758013440000000000000000000000000000000000000000000000000000000000000000000000e29"),
cpp_dec_float("6.338253001141147007483516026880000000000000000000000000000000000000000000000000000000000000000000000e29"),
cpp_dec_float("1.267650600228229401496703205376000000000000000000000000000000000000000000000000000000000000000000000e30"),
cpp_dec_float("2.535301200456458802993406410752000000000000000000000000000000000000000000000000000000000000000000000e30"),
cpp_dec_float("5.070602400912917605986812821504000000000000000000000000000000000000000000000000000000000000000000000e30"),
cpp_dec_float("1.014120480182583521197362564300800000000000000000000000000000000000000000000000000000000000000000000e31"),
cpp_dec_float("2.028240960365167042394725128601600000000000000000000000000000000000000000000000000000000000000000000e31"),
cpp_dec_float("4.056481920730334084789450257203200000000000000000000000000000000000000000000000000000000000000000000e31"),
cpp_dec_float("8.112963841460668169578900514406400000000000000000000000000000000000000000000000000000000000000000000e31"),
cpp_dec_float("1.622592768292133633915780102881280000000000000000000000000000000000000000000000000000000000000000000e32"),
cpp_dec_float("3.245185536584267267831560205762560000000000000000000000000000000000000000000000000000000000000000000e32"),
cpp_dec_float("6.490371073168534535663120411525120000000000000000000000000000000000000000000000000000000000000000000e32"),
cpp_dec_float("1.298074214633706907132624082305024000000000000000000000000000000000000000000000000000000000000000000e33"),
cpp_dec_float("2.596148429267413814265248164610048000000000000000000000000000000000000000000000000000000000000000000e33"),
cpp_dec_float("5.192296858534827628530496329220096000000000000000000000000000000000000000000000000000000000000000000e33"),
cpp_dec_float("1.038459371706965525706099265844019200000000000000000000000000000000000000000000000000000000000000000e34"),
cpp_dec_float("2.076918743413931051412198531688038400000000000000000000000000000000000000000000000000000000000000000e34"),
cpp_dec_float("4.153837486827862102824397063376076800000000000000000000000000000000000000000000000000000000000000000e34"),
cpp_dec_float("8.307674973655724205648794126752153600000000000000000000000000000000000000000000000000000000000000000e34"),
cpp_dec_float("1.661534994731144841129758825350430720000000000000000000000000000000000000000000000000000000000000000e35"),
cpp_dec_float("3.323069989462289682259517650700861440000000000000000000000000000000000000000000000000000000000000000e35"),
cpp_dec_float("6.646139978924579364519035301401722880000000000000000000000000000000000000000000000000000000000000000e35"),
cpp_dec_float("1.329227995784915872903807060280344576000000000000000000000000000000000000000000000000000000000000000e36"),
cpp_dec_float("2.658455991569831745807614120560689152000000000000000000000000000000000000000000000000000000000000000e36"),
cpp_dec_float("5.316911983139663491615228241121378304000000000000000000000000000000000000000000000000000000000000000e36"),
cpp_dec_float("1.063382396627932698323045648224275660800000000000000000000000000000000000000000000000000000000000000e37"),
cpp_dec_float("2.126764793255865396646091296448551321600000000000000000000000000000000000000000000000000000000000000e37"),
cpp_dec_float("4.253529586511730793292182592897102643200000000000000000000000000000000000000000000000000000000000000e37"),
cpp_dec_float("8.507059173023461586584365185794205286400000000000000000000000000000000000000000000000000000000000000e37"),
cpp_dec_float("1.701411834604692317316873037158841057280000000000000000000000000000000000000000000000000000000000000e38")}};

if ((p > static_cast<boost::long_long_type>(-128)) && (p < static_cast<boost::long_long_type>(+128)))
{
return p2_data[static_cast<std::size_t>(p + ((p2_data.size() - 1u) / 2u))];
}
else
{
if (p < static_cast<boost::long_long_type>(0))
{
return pow2(static_cast<boost::long_long_type>(-p)).calculate_inv();
}
else
{
cpp_dec_float<Digits10, ExponentType, Allocator> t;
default_ops::detail::pow_imp(t, two(), p, mpl::true_());
return t;
}
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_add(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& o)
{
result += o;
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_subtract(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& o)
{
result -= o;
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_multiply(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& o)
{
result *= o;
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_divide(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& o)
{
result /= o;
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_add(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const boost::ulong_long_type& o)
{
result.add_unsigned_long_long(o);
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_subtract(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const boost::ulong_long_type& o)
{
result.sub_unsigned_long_long(o);
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_multiply(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const boost::ulong_long_type& o)
{
result.mul_unsigned_long_long(o);
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_divide(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const boost::ulong_long_type& o)
{
result.div_unsigned_long_long(o);
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_add(cpp_dec_float<Digits10, ExponentType, Allocator>& result, boost::long_long_type o)
{
if (o < 0)
result.sub_unsigned_long_long(boost::multiprecision::detail::unsigned_abs(o));
else
result.add_unsigned_long_long(o);
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_subtract(cpp_dec_float<Digits10, ExponentType, Allocator>& result, boost::long_long_type o)
{
if (o < 0)
result.add_unsigned_long_long(boost::multiprecision::detail::unsigned_abs(o));
else
result.sub_unsigned_long_long(o);
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_multiply(cpp_dec_float<Digits10, ExponentType, Allocator>& result, boost::long_long_type o)
{
if (o < 0)
{
result.mul_unsigned_long_long(boost::multiprecision::detail::unsigned_abs(o));
result.negate();
}
else
result.mul_unsigned_long_long(o);
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_divide(cpp_dec_float<Digits10, ExponentType, Allocator>& result, boost::long_long_type o)
{
if (o < 0)
{
result.div_unsigned_long_long(boost::multiprecision::detail::unsigned_abs(o));
result.negate();
}
else
result.div_unsigned_long_long(o);
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_convert_to(boost::ulong_long_type* result, const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
*result = val.extract_unsigned_long_long();
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_convert_to(boost::long_long_type* result, const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
*result = val.extract_signed_long_long();
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_convert_to(long double* result, const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
*result = val.extract_long_double();
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_convert_to(double* result, const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
*result = val.extract_double();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline int eval_fpclassify(const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
if ((x.isinf)())
return FP_INFINITE;
if ((x.isnan)())
return FP_NAN;
if (x.iszero())
return FP_ZERO;
return FP_NORMAL;
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_abs(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
result = x;
if (x.isneg())
result.negate();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_fabs(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
result = x;
if (x.isneg())
result.negate();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_sqrt(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
result = x;
result.calculate_sqrt();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_floor(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
result = x;
if (!(x.isfinite)() || x.isint())
{
if ((x.isnan)())
errno = EDOM;
return;
}

if (x.isneg())
result -= cpp_dec_float<Digits10, ExponentType, Allocator>::one();
result = result.extract_integer_part();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_ceil(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
result = x;
if (!(x.isfinite)() || x.isint())
{
if ((x.isnan)())
errno = EDOM;
return;
}

if (!x.isneg())
result += cpp_dec_float<Digits10, ExponentType, Allocator>::one();
result = result.extract_integer_part();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_trunc(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x)
{
if (x.isint() || !(x.isfinite)())
{
result = x;
if ((x.isnan)())
errno = EDOM;
return;
}
result = x.extract_integer_part();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline ExponentType eval_ilogb(const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
if (val.iszero())
return (std::numeric_limits<ExponentType>::min)();
if ((val.isinf)())
return INT_MAX;
if ((val.isnan)())
#ifdef FP_ILOGBNAN
return FP_ILOGBNAN;
#else
return INT_MAX;
#endif
return val.order();
}
template <unsigned Digits10, class ExponentType, class Allocator, class ArgType>
inline void eval_scalbn(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& val, ArgType e_)
{
using default_ops::eval_multiply;
const ExponentType                               e = static_cast<ExponentType>(e_);
cpp_dec_float<Digits10, ExponentType, Allocator> t(1.0, e);
eval_multiply(result, val, t);
}

template <unsigned Digits10, class ExponentType, class Allocator, class ArgType>
inline void eval_ldexp(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x, ArgType e)
{
const boost::long_long_type the_exp = static_cast<boost::long_long_type>(e);

if ((the_exp > (std::numeric_limits<ExponentType>::max)()) || (the_exp < (std::numeric_limits<ExponentType>::min)()))
BOOST_THROW_EXCEPTION(std::runtime_error(std::string("Exponent value is out of range.")));

result = x;

if ((the_exp > static_cast<boost::long_long_type>(-std::numeric_limits<boost::long_long_type>::digits)) && (the_exp < static_cast<boost::long_long_type>(0)))
result.div_unsigned_long_long(1ULL << static_cast<boost::long_long_type>(-the_exp));
else if ((the_exp < static_cast<boost::long_long_type>(std::numeric_limits<boost::long_long_type>::digits)) && (the_exp > static_cast<boost::long_long_type>(0)))
result.mul_unsigned_long_long(1ULL << the_exp);
else if (the_exp != static_cast<boost::long_long_type>(0))
{
if ((the_exp < cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_min_exp / 2) && (x.order() > 0))
{
boost::long_long_type half_exp = e / 2;
cpp_dec_float<Digits10, ExponentType, Allocator> t = cpp_dec_float<Digits10, ExponentType, Allocator>::pow2(half_exp);
result *= t;
if (2 * half_exp != e)
t *= 2;
result *= t;
}
else
result *= cpp_dec_float<Digits10, ExponentType, Allocator>::pow2(e);
}
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline void eval_frexp(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x, ExponentType* e)
{
result = x;

if (result.iszero() || (result.isinf)() || (result.isnan)())
{
*e = 0;
return;
}

if (result.isneg())
result.negate();

ExponentType t = result.order();
BOOST_MP_USING_ABS
if (abs(t) < ((std::numeric_limits<ExponentType>::max)() / 1000))
{
t *= 1000;
t /= 301;
}
else
{
t /= 301;
t *= 1000;
}

result *= cpp_dec_float<Digits10, ExponentType, Allocator>::pow2(-t);

if (result.iszero() || (result.isinf)() || (result.isnan)())
{
result = x;
if (result.isneg())
result.negate();
t /= 2;
result *= cpp_dec_float<Digits10, ExponentType, Allocator>::pow2(-t);
}
BOOST_MP_USING_ABS
if (abs(result.order()) > 5)
{
ExponentType                                     e2;
cpp_dec_float<Digits10, ExponentType, Allocator> r2;
eval_frexp(r2, result, &e2);
if ((t > 0) && (e2 > 0) && (t > (std::numeric_limits<ExponentType>::max)() - e2))
BOOST_THROW_EXCEPTION(std::runtime_error("Exponent is too large to be represented as a power of 2."));
if ((t < 0) && (e2 < 0) && (t < (std::numeric_limits<ExponentType>::min)() - e2))
BOOST_THROW_EXCEPTION(std::runtime_error("Exponent is too large to be represented as a power of 2."));
t += e2;
result = r2;
}

while (result.compare(cpp_dec_float<Digits10, ExponentType, Allocator>::one()) >= 0)
{
result /= cpp_dec_float<Digits10, ExponentType, Allocator>::two();
++t;
}
while (result.compare(cpp_dec_float<Digits10, ExponentType, Allocator>::half()) < 0)
{
result *= cpp_dec_float<Digits10, ExponentType, Allocator>::two();
--t;
}
*e = t;
if (x.isneg())
result.negate();
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline typename disable_if<is_same<ExponentType, int> >::type eval_frexp(cpp_dec_float<Digits10, ExponentType, Allocator>& result, const cpp_dec_float<Digits10, ExponentType, Allocator>& x, int* e)
{
ExponentType t;
eval_frexp(result, x, &t);
if ((t > (std::numeric_limits<int>::max)()) || (t < (std::numeric_limits<int>::min)()))
BOOST_THROW_EXCEPTION(std::runtime_error("Exponent is outside the range of an int"));
*e = static_cast<int>(t);
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline bool eval_is_zero(const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
return val.iszero();
}
template <unsigned Digits10, class ExponentType, class Allocator>
inline int eval_get_sign(const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
return val.iszero() ? 0 : val.isneg() ? -1 : 1;
}

template <unsigned Digits10, class ExponentType, class Allocator>
inline std::size_t hash_value(const cpp_dec_float<Digits10, ExponentType, Allocator>& val)
{
return val.hash();
}

} 

using boost::multiprecision::backends::cpp_dec_float;

typedef number<cpp_dec_float<50> >  cpp_dec_float_50;
typedef number<cpp_dec_float<100> > cpp_dec_float_100;

namespace detail {

template <unsigned Digits10, class ExponentType, class Allocator>
struct transcendental_reduction_type<boost::multiprecision::backends::cpp_dec_float<Digits10, ExponentType, Allocator> >
{
typedef boost::multiprecision::backends::cpp_dec_float<Digits10 * 3, ExponentType, Allocator> type;
};

#ifdef BOOST_NO_SFINAE_EXPR

template <unsigned D1, class E1, class A1, unsigned D2, class E2, class A2>
struct is_explicitly_convertible<cpp_dec_float<D1, E1, A1>, cpp_dec_float<D2, E2, A2> > : public mpl::true_
{};

#endif

} 


}} 

namespace std {
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
class numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >
{
public:
BOOST_STATIC_CONSTEXPR bool is_specialized                      = true;
BOOST_STATIC_CONSTEXPR bool is_signed                           = true;
BOOST_STATIC_CONSTEXPR bool is_integer                          = false;
BOOST_STATIC_CONSTEXPR bool is_exact                            = false;
BOOST_STATIC_CONSTEXPR bool is_bounded                          = true;
BOOST_STATIC_CONSTEXPR bool is_modulo                           = false;
BOOST_STATIC_CONSTEXPR bool is_iec559                           = false;
BOOST_STATIC_CONSTEXPR int  digits                              = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10;
BOOST_STATIC_CONSTEXPR int  digits10                            = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10;
BOOST_STATIC_CONSTEXPR int  max_digits10                        = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_total_digits10;
BOOST_STATIC_CONSTEXPR ExponentType min_exponent                = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_min_exp;   
BOOST_STATIC_CONSTEXPR ExponentType min_exponent10              = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_min_exp10; 
BOOST_STATIC_CONSTEXPR ExponentType max_exponent                = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_max_exp;   
BOOST_STATIC_CONSTEXPR ExponentType max_exponent10              = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_max_exp10; 
BOOST_STATIC_CONSTEXPR int          radix                       = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_radix;
BOOST_STATIC_CONSTEXPR std::float_round_style round_style       = std::round_indeterminate;
BOOST_STATIC_CONSTEXPR bool                   has_infinity      = true;
BOOST_STATIC_CONSTEXPR bool                   has_quiet_NaN     = true;
BOOST_STATIC_CONSTEXPR bool                   has_signaling_NaN = false;
BOOST_STATIC_CONSTEXPR std::float_denorm_style has_denorm       = std::denorm_absent;
BOOST_STATIC_CONSTEXPR bool                    has_denorm_loss  = false;
BOOST_STATIC_CONSTEXPR bool                    traps            = false;
BOOST_STATIC_CONSTEXPR bool                    tinyness_before  = false;

BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates>(min)() { return (boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::min)(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates>(max)() { return (boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::max)(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> lowest() { return boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::zero(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> epsilon() { return boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::eps(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> round_error() { return 0.5L; }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> infinity() { return boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::inf(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> quiet_NaN() { return boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::nan(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> signaling_NaN() { return boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::zero(); }
BOOST_STATIC_CONSTEXPR boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> denorm_min() { return boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::zero(); }
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION

template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST int numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::digits;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST int numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::digits10;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST int numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::max_digits10;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::is_signed;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::is_integer;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::is_exact;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST int numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::radix;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST ExponentType numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::min_exponent;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST ExponentType numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::min_exponent10;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST ExponentType numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::max_exponent;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST ExponentType numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::max_exponent10;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::has_infinity;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::has_quiet_NaN;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::has_signaling_NaN;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST float_denorm_style numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::has_denorm;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::has_denorm_loss;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::is_iec559;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::is_bounded;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::is_modulo;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::traps;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST bool numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::tinyness_before;
template <unsigned Digits10, class ExponentType, class Allocator, boost::multiprecision::expression_template_option ExpressionTemplates>
BOOST_CONSTEXPR_OR_CONST float_round_style numeric_limits<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates> >::round_style;

#endif
} 

namespace boost {
namespace math {

namespace policies {

template <unsigned Digits10, class ExponentType, class Allocator, class Policy, boost::multiprecision::expression_template_option ExpressionTemplates>
struct precision<boost::multiprecision::number<boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>, ExpressionTemplates>, Policy>
{
static const boost::int32_t cpp_dec_float_digits10 = boost::multiprecision::cpp_dec_float<Digits10, ExponentType, Allocator>::cpp_dec_float_digits10;

typedef typename Policy::precision_type                            precision_type;
typedef digits2<((cpp_dec_float_digits10 + 1LL) * 1000LL) / 301LL> digits_2;
typedef typename mpl::if_c<
((digits_2::value <= precision_type::value) || (Policy::precision_type::value <= 0)),
digits_2,
precision_type>::type type;
};

}

}} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
