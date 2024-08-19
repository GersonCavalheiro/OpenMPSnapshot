


#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 6326) 
#endif

namespace detail {

template <typename T, typename U>
inline void pow_imp(T& result, const T& t, const U& p, const mpl::false_&)
{

typedef typename boost::multiprecision::detail::canonical<U, T>::type int_type;

if (&result == &t)
{
T temp;
pow_imp(temp, t, p, mpl::false_());
result = temp;
return;
}

if (U(p % U(2)) != U(0))
{
result = t;
}
else
result = int_type(1);

U p2(p);

T x(t);

while (U(p2 /= 2) != U(0))
{
eval_multiply(x, x);

const bool has_binary_power = (U(p2 % U(2)) != U(0));

if (has_binary_power)
{
eval_multiply(result, x);
}
}
}

template <typename T, typename U>
inline void pow_imp(T& result, const T& t, const U& p, const mpl::true_&)
{
typedef typename boost::multiprecision::detail::canonical<U, T>::type int_type;
typedef typename make_unsigned<U>::type                               ui_type;

if (p < 0)
{
T temp;
temp = static_cast<int_type>(1);
T denom;
pow_imp(denom, t, static_cast<ui_type>(-p), mpl::false_());
eval_divide(result, temp, denom);
return;
}
pow_imp(result, t, static_cast<ui_type>(p), mpl::false_());
}

} 

template <typename T, typename U>
inline typename enable_if_c<is_integral<U>::value>::type eval_pow(T& result, const T& t, const U& p)
{
detail::pow_imp(result, t, p, boost::is_signed<U>());
}

template <class T>
void hyp0F0(T& H0F0, const T& x)
{

typedef typename mpl::front<typename T::unsigned_types>::type ui_type;

BOOST_ASSERT(&H0F0 != &x);
long tol = boost::multiprecision::detail::digits2<number<T, et_on> >::value();
T    t;

T x_pow_n_div_n_fact(x);

eval_add(H0F0, x_pow_n_div_n_fact, ui_type(1));

T lim;
eval_ldexp(lim, H0F0, 1 - tol);
if (eval_get_sign(lim) < 0)
lim.negate();

ui_type n;

const unsigned series_limit =
boost::multiprecision::detail::digits2<number<T, et_on> >::value() < 100
? 100
: boost::multiprecision::detail::digits2<number<T, et_on> >::value();
for (n = 2; n < series_limit; ++n)
{
eval_multiply(x_pow_n_div_n_fact, x);
eval_divide(x_pow_n_div_n_fact, n);
eval_add(H0F0, x_pow_n_div_n_fact);
bool neg = eval_get_sign(x_pow_n_div_n_fact) < 0;
if (neg)
x_pow_n_div_n_fact.negate();
if (lim.compare(x_pow_n_div_n_fact) > 0)
break;
if (neg)
x_pow_n_div_n_fact.negate();
}
if (n >= series_limit)
BOOST_THROW_EXCEPTION(std::runtime_error("H0F0 failed to converge"));
}

template <class T>
void hyp1F0(T& H1F0, const T& a, const T& x)
{

typedef typename boost::multiprecision::detail::canonical<int, T>::type si_type;

BOOST_ASSERT(&H1F0 != &x);
BOOST_ASSERT(&H1F0 != &a);

T x_pow_n_div_n_fact(x);
T pochham_a(a);
T ap(a);

eval_multiply(H1F0, pochham_a, x_pow_n_div_n_fact);
eval_add(H1F0, si_type(1));
T lim;
eval_ldexp(lim, H1F0, 1 - boost::multiprecision::detail::digits2<number<T, et_on> >::value());
if (eval_get_sign(lim) < 0)
lim.negate();

si_type n;
T       term, part;

const si_type series_limit =
boost::multiprecision::detail::digits2<number<T, et_on> >::value() < 100
? 100
: boost::multiprecision::detail::digits2<number<T, et_on> >::value();
for (n = 2; n < series_limit; n++)
{
eval_multiply(x_pow_n_div_n_fact, x);
eval_divide(x_pow_n_div_n_fact, n);
eval_increment(ap);
eval_multiply(pochham_a, ap);
eval_multiply(term, pochham_a, x_pow_n_div_n_fact);
eval_add(H1F0, term);
if (eval_get_sign(term) < 0)
term.negate();
if (lim.compare(term) >= 0)
break;
}
if (n >= series_limit)
BOOST_THROW_EXCEPTION(std::runtime_error("H1F0 failed to converge"));
}

template <class T>
void eval_exp(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The exp function is only valid for floating point types.");
if (&x == &result)
{
T temp;
eval_exp(temp, x);
result = temp;
return;
}
typedef typename boost::multiprecision::detail::canonical<unsigned, T>::type ui_type;
typedef typename boost::multiprecision::detail::canonical<int, T>::type      si_type;
typedef typename T::exponent_type                                            exp_type;
typedef typename boost::multiprecision::detail::canonical<exp_type, T>::type canonical_exp_type;

int  type  = eval_fpclassify(x);
bool isneg = eval_get_sign(x) < 0;
if (type == (int)FP_NAN)
{
result = x;
errno  = EDOM;
return;
}
else if (type == (int)FP_INFINITE)
{
if (isneg)
result = ui_type(0u);
else
result = x;
return;
}
else if (type == (int)FP_ZERO)
{
result = ui_type(1);
return;
}

T xx = x;
T exp_series;
if (isneg)
xx.negate();

if (xx.compare(si_type(1)) <= 0)
{
T lim;
if (std::numeric_limits<number<T, et_on> >::is_specialized)
lim = std::numeric_limits<number<T, et_on> >::epsilon().backend();
else
{
result = ui_type(1);
eval_ldexp(lim, result, 1 - boost::multiprecision::detail::digits2<number<T, et_on> >::value());
}
unsigned k = 2;
exp_series = xx;
result     = si_type(1);
if (isneg)
eval_subtract(result, exp_series);
else
eval_add(result, exp_series);
eval_multiply(exp_series, xx);
eval_divide(exp_series, ui_type(k));
eval_add(result, exp_series);
while (exp_series.compare(lim) > 0)
{
++k;
eval_multiply(exp_series, xx);
eval_divide(exp_series, ui_type(k));
if (isneg && (k & 1))
eval_subtract(result, exp_series);
else
eval_add(result, exp_series);
}
return;
}

typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type ll;
eval_trunc(exp_series, x);
eval_convert_to(&ll, exp_series);
if (x.compare(ll) == 0)
{
detail::pow_imp(result, get_constant_e<T>(), ll, mpl::true_());
return;
}
else if (exp_series.compare(x) == 0)
{
if (isneg)
result = ui_type(0);
else
result = std::numeric_limits<number<T> >::has_infinity ? std::numeric_limits<number<T> >::infinity().backend() : (std::numeric_limits<number<T> >::max)().backend();
return;
}



eval_divide(result, xx, get_constant_ln2<T>());
exp_type n;
eval_convert_to(&n, result);

if (n == (std::numeric_limits<exp_type>::max)())
{
if (isneg)
result = ui_type(0);
else
result = std::numeric_limits<number<T> >::has_infinity ? std::numeric_limits<number<T> >::infinity().backend() : (std::numeric_limits<number<T> >::max)().backend();
return;
}

const si_type p2 = static_cast<si_type>(si_type(1) << 11);

eval_multiply(exp_series, get_constant_ln2<T>(), static_cast<canonical_exp_type>(n));
eval_subtract(exp_series, xx);
eval_divide(exp_series, p2);
exp_series.negate();
hyp0F0(result, exp_series);

detail::pow_imp(exp_series, result, p2, mpl::true_());
result = ui_type(1);
eval_ldexp(result, result, n);
eval_multiply(exp_series, result);

if (isneg)
eval_divide(result, ui_type(1), exp_series);
else
result = exp_series;
}

template <class T>
void eval_log(T& result, const T& arg)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The log function is only valid for floating point types.");
typedef typename boost::multiprecision::detail::canonical<unsigned, T>::type ui_type;
typedef typename T::exponent_type                                            exp_type;
typedef typename boost::multiprecision::detail::canonical<exp_type, T>::type canonical_exp_type;
typedef typename mpl::front<typename T::float_types>::type                   fp_type;
int                                                                          s = eval_signbit(arg);
switch (eval_fpclassify(arg))
{
case FP_NAN:
result = arg;
errno  = EDOM;
return;
case FP_INFINITE:
if (s)
break;
result = arg;
return;
case FP_ZERO:
result = std::numeric_limits<number<T> >::has_infinity ? std::numeric_limits<number<T> >::infinity().backend() : (std::numeric_limits<number<T> >::max)().backend();
result.negate();
errno = ERANGE;
return;
}
if (s)
{
result = std::numeric_limits<number<T> >::quiet_NaN().backend();
errno  = EDOM;
return;
}

exp_type e;
T        t;
eval_frexp(t, arg, &e);
bool alternate = false;

if (t.compare(fp_type(2) / fp_type(3)) <= 0)
{
alternate = true;
eval_ldexp(t, t, 1);
--e;
}

eval_multiply(result, get_constant_ln2<T>(), canonical_exp_type(e));
INSTRUMENT_BACKEND(result);
eval_subtract(t, ui_type(1)); 
if (!alternate)
t.negate(); 
T pow = t;
T lim;
T t2;

if (alternate)
eval_add(result, t);
else
eval_subtract(result, t);

if (std::numeric_limits<number<T, et_on> >::is_specialized)
eval_multiply(lim, result, std::numeric_limits<number<T, et_on> >::epsilon().backend());
else
eval_ldexp(lim, result, 1 - boost::multiprecision::detail::digits2<number<T, et_on> >::value());
if (eval_get_sign(lim) < 0)
lim.negate();
INSTRUMENT_BACKEND(lim);

ui_type k = 1;
do
{
++k;
eval_multiply(pow, t);
eval_divide(t2, pow, k);
INSTRUMENT_BACKEND(t2);
if (alternate && ((k & 1) != 0))
eval_add(result, t2);
else
eval_subtract(result, t2);
INSTRUMENT_BACKEND(result);
} while (lim.compare(t2) < 0);
}

template <class T>
const T& get_constant_log10()
{
static BOOST_MP_THREAD_LOCAL T    result;
static BOOST_MP_THREAD_LOCAL long digits = 0;
#ifndef BOOST_MP_USING_THREAD_LOCAL
static BOOST_MP_THREAD_LOCAL bool b = false;
constant_initializer<T, &get_constant_log10<T> >::do_nothing();

if (!b || (digits != boost::multiprecision::detail::digits2<number<T> >::value()))
{
b = true;
#else
if ((digits != boost::multiprecision::detail::digits2<number<T> >::value()))
{
#endif
typedef typename boost::multiprecision::detail::canonical<unsigned, T>::type ui_type;
T                                                                            ten;
ten = ui_type(10u);
eval_log(result, ten);
digits = boost::multiprecision::detail::digits2<number<T> >::value();
}

return result;
}

template <class T>
void eval_log10(T& result, const T& arg)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The log10 function is only valid for floating point types.");
eval_log(result, arg);
eval_divide(result, get_constant_log10<T>());
}

template <class R, class T>
inline void eval_log2(R& result, const T& a)
{
eval_log(result, a);
eval_divide(result, get_constant_ln2<R>());
}

template <typename T>
inline void eval_pow(T& result, const T& x, const T& a)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The pow function is only valid for floating point types.");
typedef typename boost::multiprecision::detail::canonical<int, T>::type si_type;
typedef typename mpl::front<typename T::float_types>::type              fp_type;

if ((&result == &x) || (&result == &a))
{
T t;
eval_pow(t, x, a);
result = t;
return;
}

if ((a.compare(si_type(1)) == 0) || (x.compare(si_type(1)) == 0))
{
result = x;
return;
}
if (a.compare(si_type(0)) == 0)
{
result = si_type(1);
return;
}

int type = eval_fpclassify(x);

switch (type)
{
case FP_ZERO:
switch (eval_fpclassify(a))
{
case FP_ZERO:
result = si_type(1);
break;
case FP_NAN:
result = a;
break;
case FP_NORMAL:
{
try
{
typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type i;
eval_convert_to(&i, a);
if (a.compare(i) == 0)
{
if (eval_signbit(a))
{
if (i & 1)
{
result = std::numeric_limits<number<T> >::infinity().backend();
if (eval_signbit(x))
result.negate();
errno = ERANGE;
}
else
{
result = std::numeric_limits<number<T> >::infinity().backend();
errno  = ERANGE;
}
}
else if (i & 1)
{
result = x;
}
else
result = si_type(0);
return;
}
}
catch (const std::exception&)
{
}
BOOST_FALLTHROUGH;
}
default:
if (eval_signbit(a))
{
result = std::numeric_limits<number<T> >::infinity().backend();
errno  = ERANGE;
}
else
result = x;
break;
}
return;
case FP_NAN:
result = x;
errno  = ERANGE;
return;
default:;
}

int s = eval_get_sign(a);
if (s == 0)
{
result = si_type(1);
return;
}

if (s < 0)
{
T t, da;
t = a;
t.negate();
eval_pow(da, x, t);
eval_divide(result, si_type(1), da);
return;
}

typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type an;
typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type max_an =
std::numeric_limits<typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type>::is_specialized ? (std::numeric_limits<typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type>::max)() : static_cast<typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type>(1) << (sizeof(typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type) * CHAR_BIT - 2);
typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type min_an =
std::numeric_limits<typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type>::is_specialized ? (std::numeric_limits<typename boost::multiprecision::detail::canonical<boost::intmax_t, T>::type>::min)() : -min_an;

T fa;
#ifndef BOOST_NO_EXCEPTIONS
try
{
#endif
eval_convert_to(&an, a);
if (a.compare(an) == 0)
{
detail::pow_imp(result, x, an, mpl::true_());
return;
}
#ifndef BOOST_NO_EXCEPTIONS
}
catch (const std::exception&)
{
an = (std::numeric_limits<boost::intmax_t>::max)();
}
#endif
if ((eval_get_sign(x) < 0))
{
typename boost::multiprecision::detail::canonical<boost::uintmax_t, T>::type aun;
#ifndef BOOST_NO_EXCEPTIONS
try
{
#endif
eval_convert_to(&aun, a);
if (a.compare(aun) == 0)
{
fa = x;
fa.negate();
eval_pow(result, fa, a);
if (aun & 1u)
result.negate();
return;
}
#ifndef BOOST_NO_EXCEPTIONS
}
catch (const std::exception&)
{
}
#endif
eval_floor(result, a);
if ((x.compare(si_type(-1)) == 0) && (eval_fpclassify(a) == FP_INFINITE))
{
result = si_type(1);
}
else if (a.compare(result) == 0)
{
if (x.compare(si_type(-1)) < 0)
{
result = std::numeric_limits<number<T, et_on> >::infinity().backend();
}
else
{
result = si_type(0);
}
}
else if (type == FP_INFINITE)
{
result = std::numeric_limits<number<T, et_on> >::infinity().backend();
}
else if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
{
BOOST_THROW_EXCEPTION(std::domain_error("Result of pow is undefined or non-real and there is no NaN for this number type."));
}
return;
}

T t, da;

eval_subtract(da, a, an);

if ((x.compare(fp_type(0.5)) >= 0) && (x.compare(fp_type(0.9)) < 0) && (an < max_an) && (an > min_an))
{
if (a.compare(fp_type(1e-5f)) <= 0)
{
eval_log(t, x);
eval_multiply(t, a);
hyp0F0(result, t);
return;
}
else
{
if (an)
{
da.negate();
t = si_type(1);
eval_subtract(t, x);
hyp1F0(result, da, t);
detail::pow_imp(t, x, an, mpl::true_());
eval_multiply(result, t);
}
else
{
da = a;
da.negate();
t = si_type(1);
eval_subtract(t, x);
hyp1F0(result, da, t);
}
}
}
else
{
if (an)
{
eval_log(t, x);
eval_multiply(t, da);
eval_exp(result, t);
detail::pow_imp(t, x, an, mpl::true_());
eval_multiply(result, t);
}
else
{
eval_log(t, x);
eval_multiply(t, a);
eval_exp(result, t);
}
}
}

template <class T, class A>
#if BOOST_WORKAROUND(BOOST_MSVC, < 1800)
inline typename enable_if_c<!is_integral<A>::value, void>::type
#else
inline typename enable_if_c<is_compatible_arithmetic_type<A, number<T> >::value && !is_integral<A>::value, void>::type
#endif
eval_pow(T& result, const T& x, const A& a)
{
typedef typename boost::multiprecision::detail::canonical<A, T>::type          canonical_type;
typedef typename mpl::if_<is_same<A, canonical_type>, T, canonical_type>::type cast_type;
cast_type                                                                      c;
c = a;
eval_pow(result, x, c);
}

template <class T, class A>
#if BOOST_WORKAROUND(BOOST_MSVC, < 1800)
inline void
#else
inline typename enable_if_c<is_compatible_arithmetic_type<A, number<T> >::value, void>::type
#endif
eval_pow(T& result, const A& x, const T& a)
{
typedef typename boost::multiprecision::detail::canonical<A, T>::type          canonical_type;
typedef typename mpl::if_<is_same<A, canonical_type>, T, canonical_type>::type cast_type;
cast_type                                                                      c;
c = x;
eval_pow(result, c, a);
}

template <class T>
void eval_exp2(T& result, const T& arg)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The log function is only valid for floating point types.");

typename boost::multiprecision::detail::canonical<typename T::exponent_type, T>::type i;
T                                                                                     temp;
try
{
eval_trunc(temp, arg);
eval_convert_to(&i, temp);
if (arg.compare(i) == 0)
{
temp = static_cast<typename mpl::front<typename T::unsigned_types>::type>(1u);
eval_ldexp(result, temp, i);
return;
}
}
catch (const boost::math::rounding_error&)
{ 
}
catch (const std::runtime_error&)
{ 
}

temp = static_cast<typename mpl::front<typename T::unsigned_types>::type>(2u);
eval_pow(result, temp, arg);
}

namespace detail {

template <class T>
void small_sinh_series(T x, T& result)
{
typedef typename boost::multiprecision::detail::canonical<unsigned, T>::type ui_type;
bool                                                                         neg = eval_get_sign(x) < 0;
if (neg)
x.negate();
T p(x);
T mult(x);
eval_multiply(mult, x);
result    = x;
ui_type k = 1;

T lim(x);
eval_ldexp(lim, lim, 1 - boost::multiprecision::detail::digits2<number<T, et_on> >::value());

do
{
eval_multiply(p, mult);
eval_divide(p, ++k);
eval_divide(p, ++k);
eval_add(result, p);
} while (p.compare(lim) >= 0);
if (neg)
result.negate();
}

template <class T>
void sinhcosh(const T& x, T* p_sinh, T* p_cosh)
{
typedef typename boost::multiprecision::detail::canonical<unsigned, T>::type ui_type;
typedef typename mpl::front<typename T::float_types>::type                   fp_type;

switch (eval_fpclassify(x))
{
case FP_NAN:
errno = EDOM;
case FP_INFINITE:
if (p_sinh)
*p_sinh = x;
if (p_cosh)
{
*p_cosh = x;
if (eval_get_sign(x) < 0)
p_cosh->negate();
}
return;
case FP_ZERO:
if (p_sinh)
*p_sinh = x;
if (p_cosh)
*p_cosh = ui_type(1);
return;
default:;
}

bool small_sinh = eval_get_sign(x) < 0 ? x.compare(fp_type(-0.5)) > 0 : x.compare(fp_type(0.5)) < 0;

if (p_cosh || !small_sinh)
{
T e_px, e_mx;
eval_exp(e_px, x);
eval_divide(e_mx, ui_type(1), e_px);
if (eval_signbit(e_mx) != eval_signbit(e_px))
e_mx.negate(); 

if (p_sinh)
{
if (small_sinh)
{
small_sinh_series(x, *p_sinh);
}
else
{
eval_subtract(*p_sinh, e_px, e_mx);
eval_ldexp(*p_sinh, *p_sinh, -1);
}
}
if (p_cosh)
{
eval_add(*p_cosh, e_px, e_mx);
eval_ldexp(*p_cosh, *p_cosh, -1);
}
}
else
{
small_sinh_series(x, *p_sinh);
}
}

} 

template <class T>
inline void eval_sinh(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The sinh function is only valid for floating point types.");
detail::sinhcosh(x, &result, static_cast<T*>(0));
}

template <class T>
inline void eval_cosh(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The cosh function is only valid for floating point types.");
detail::sinhcosh(x, static_cast<T*>(0), &result);
}

template <class T>
inline void eval_tanh(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The tanh function is only valid for floating point types.");
T c;
detail::sinhcosh(x, &result, &c);
if ((eval_fpclassify(result) == FP_INFINITE) && (eval_fpclassify(c) == FP_INFINITE))
{
bool s = eval_signbit(result) != eval_signbit(c);
result = static_cast<typename mpl::front<typename T::unsigned_types>::type>(1u);
if (s)
result.negate();
return;
}
eval_divide(result, c);
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
