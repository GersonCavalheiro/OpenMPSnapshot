


#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable : 6326) 
#endif

template <class T>
void hyp0F1(T& result, const T& b, const T& x)
{
typedef typename boost::multiprecision::detail::canonical<boost::int32_t, T>::type  si_type;
typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;


T x_pow_n_div_n_fact(x);
T pochham_b(b);
T bp(b);

eval_divide(result, x_pow_n_div_n_fact, pochham_b);
eval_add(result, ui_type(1));

si_type n;

T tol;
tol = ui_type(1);
eval_ldexp(tol, tol, 1 - boost::multiprecision::detail::digits2<number<T, et_on> >::value());
eval_multiply(tol, result);
if (eval_get_sign(tol) < 0)
tol.negate();
T term;

const int series_limit =
boost::multiprecision::detail::digits2<number<T, et_on> >::value() < 100
? 100
: boost::multiprecision::detail::digits2<number<T, et_on> >::value();
for (n = 2; n < series_limit; ++n)
{
eval_multiply(x_pow_n_div_n_fact, x);
eval_divide(x_pow_n_div_n_fact, n);
eval_increment(bp);
eval_multiply(pochham_b, bp);

eval_divide(term, x_pow_n_div_n_fact, pochham_b);
eval_add(result, term);

bool neg_term = eval_get_sign(term) < 0;
if (neg_term)
term.negate();
if (term.compare(tol) <= 0)
break;
}

if (n >= series_limit)
BOOST_THROW_EXCEPTION(std::runtime_error("H0F1 Failed to Converge"));
}

template <class T, unsigned N, bool b = boost::multiprecision::detail::is_variable_precision<boost::multiprecision::number<T> >::value>
struct scoped_N_precision
{
template <class U>
scoped_N_precision(U const&) {}
template <class U>
void reduce(U&) {}
};

template <class T, unsigned N>
struct scoped_N_precision<T, N, true>
{
unsigned old_precision, old_arg_precision;
scoped_N_precision(T& arg)
{
old_precision = T::default_precision();
old_arg_precision = arg.precision();
T::default_precision(old_arg_precision * N);
arg.precision(old_arg_precision * N);
}
~scoped_N_precision()
{
T::default_precision(old_precision);
}
void reduce(T& arg) 
{
arg.precision(old_arg_precision);
}
};

template <class T>
void reduce_n_half_pi(T& arg, const T& n, bool go_down)
{
typedef typename boost::multiprecision::detail::transcendental_reduction_type<T>::type reduction_type;
reduction_type big_arg(arg);
scoped_N_precision<T, 3> scoped_precision(big_arg);
reduction_type reduction = get_constant_pi<reduction_type>();
eval_ldexp(reduction, reduction, -1); 
eval_multiply(reduction, n);
BOOST_MATH_INSTRUMENT_CODE(big_arg.str(10, std::ios_base::scientific));
BOOST_MATH_INSTRUMENT_CODE(reduction.str(10, std::ios_base::scientific));

if (go_down)
eval_subtract(big_arg, reduction, big_arg);
else
eval_subtract(big_arg, reduction);
arg = T(big_arg);
scoped_precision.reduce(arg);
BOOST_MATH_INSTRUMENT_CODE(big_arg.str(10, std::ios_base::scientific));
BOOST_MATH_INSTRUMENT_CODE(arg.str(10, std::ios_base::scientific));
}

template <class T>
void eval_sin(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The sin function is only valid for floating point types.");
BOOST_MATH_INSTRUMENT_CODE(x.str(0, std::ios_base::scientific));
if (&result == &x)
{
T temp;
eval_sin(temp, x);
result = temp;
return;
}

typedef typename boost::multiprecision::detail::canonical<boost::int32_t, T>::type  si_type;
typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;
typedef typename mpl::front<typename T::float_types>::type                          fp_type;

switch (eval_fpclassify(x))
{
case FP_INFINITE:
case FP_NAN:
if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
BOOST_THROW_EXCEPTION(std::domain_error("Result is undefined or complex and there is no NaN for this number type."));
return;
case FP_ZERO:
result = x;
return;
default:;
}

T xx = x;

bool b_negate_sin = false;

if (eval_get_sign(x) < 0)
{
xx.negate();
b_negate_sin = !b_negate_sin;
}

T n_pi, t;
T half_pi = get_constant_pi<T>();
eval_ldexp(half_pi, half_pi, -1); 
if (xx.compare(half_pi) > 0)
{
eval_divide(n_pi, xx, half_pi);
eval_trunc(n_pi, n_pi);
t = ui_type(4);
eval_fmod(t, n_pi, t);
bool b_go_down = false;
if (t.compare(ui_type(1)) == 0)
{
b_go_down = true;
}
else if (t.compare(ui_type(2)) == 0)
{
b_negate_sin = !b_negate_sin;
}
else if (t.compare(ui_type(3)) == 0)
{
b_negate_sin = !b_negate_sin;
b_go_down    = true;
}

if (b_go_down)
eval_increment(n_pi);
if (n_pi.compare(get_constant_one_over_epsilon<T>()) > 0)
{
result = ui_type(0);
return;
}

reduce_n_half_pi(xx, n_pi, b_go_down);
if (eval_get_sign(xx) < 0)
{
xx.negate();
b_negate_sin = !b_negate_sin;
}
if (xx.compare(half_pi) > 0)
{
eval_ldexp(half_pi, half_pi, 1);
eval_subtract(xx, half_pi, xx);
eval_ldexp(half_pi, half_pi, -1);
b_go_down = !b_go_down;
}

BOOST_MATH_INSTRUMENT_CODE(xx.str(0, std::ios_base::scientific));
BOOST_MATH_INSTRUMENT_CODE(n_pi.str(0, std::ios_base::scientific));
BOOST_ASSERT(xx.compare(half_pi) <= 0);
BOOST_ASSERT(xx.compare(ui_type(0)) >= 0);
}

t = half_pi;
eval_subtract(t, xx);

const bool b_zero    = eval_get_sign(xx) == 0;
const bool b_pi_half = eval_get_sign(t) == 0;

BOOST_MATH_INSTRUMENT_CODE(xx.str(0, std::ios_base::scientific));
BOOST_MATH_INSTRUMENT_CODE(t.str(0, std::ios_base::scientific));

const bool b_near_zero    = xx.compare(fp_type(1e-1)) < 0;
const bool b_near_pi_half = t.compare(fp_type(1e-1)) < 0;

if (b_zero)
{
result = ui_type(0);
}
else if (b_pi_half)
{
result = ui_type(1);
}
else if (b_near_zero)
{
eval_multiply(t, xx, xx);
eval_divide(t, si_type(-4));
T t2;
t2 = fp_type(1.5);
hyp0F1(result, t2, t);
BOOST_MATH_INSTRUMENT_CODE(result.str(0, std::ios_base::scientific));
eval_multiply(result, xx);
}
else if (b_near_pi_half)
{
eval_multiply(t, t);
eval_divide(t, si_type(-4));
T t2;
t2 = fp_type(0.5);
hyp0F1(result, t2, t);
BOOST_MATH_INSTRUMENT_CODE(result.str(0, std::ios_base::scientific));
}
else
{

static const si_type n_scale           = 9;
static const si_type n_three_pow_scale = static_cast<si_type>(19683L);

eval_divide(xx, n_three_pow_scale);

eval_multiply(t, xx, xx);
eval_divide(t, si_type(-4));
T t2;
t2 = fp_type(1.5);
hyp0F1(result, t2, t);
BOOST_MATH_INSTRUMENT_CODE(result.str(0, std::ios_base::scientific));
eval_multiply(result, xx);

for (boost::int32_t k = static_cast<boost::int32_t>(0); k < n_scale; k++)
{
eval_multiply(t2, result, ui_type(3));
eval_multiply(t, result, result);
eval_multiply(t, result);
eval_multiply(t, ui_type(4));
eval_subtract(result, t2, t);
}
}

if (b_negate_sin)
result.negate();
BOOST_MATH_INSTRUMENT_CODE(result.str(0, std::ios_base::scientific));
}

template <class T>
void eval_cos(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The cos function is only valid for floating point types.");
if (&result == &x)
{
T temp;
eval_cos(temp, x);
result = temp;
return;
}

typedef typename boost::multiprecision::detail::canonical<boost::int32_t, T>::type  si_type;
typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;

switch (eval_fpclassify(x))
{
case FP_INFINITE:
case FP_NAN:
if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
BOOST_THROW_EXCEPTION(std::domain_error("Result is undefined or complex and there is no NaN for this number type."));
return;
case FP_ZERO:
result = ui_type(1);
return;
default:;
}

T xx = x;

bool b_negate_cos = false;

if (eval_get_sign(x) < 0)
{
xx.negate();
}
BOOST_MATH_INSTRUMENT_CODE(xx.str(0, std::ios_base::scientific));

T n_pi, t;
T half_pi = get_constant_pi<T>();
eval_ldexp(half_pi, half_pi, -1); 
if (xx.compare(half_pi) > 0)
{
eval_divide(t, xx, half_pi);
eval_trunc(n_pi, t);
BOOST_MATH_INSTRUMENT_CODE(n_pi.str(0, std::ios_base::scientific));
t = ui_type(4);
eval_fmod(t, n_pi, t);

bool b_go_down = false;
if (t.compare(ui_type(0)) == 0)
{
b_go_down = true;
}
else if (t.compare(ui_type(1)) == 0)
{
b_negate_cos = true;
}
else if (t.compare(ui_type(2)) == 0)
{
b_go_down    = true;
b_negate_cos = true;
}
else
{
BOOST_ASSERT(t.compare(ui_type(3)) == 0);
}

if (b_go_down)
eval_increment(n_pi);
if (n_pi.compare(get_constant_one_over_epsilon<T>()) > 0)
{
result = ui_type(1);
return;
}

reduce_n_half_pi(xx, n_pi, b_go_down);
if (eval_get_sign(xx) < 0)
{
xx.negate();
b_negate_cos = !b_negate_cos;
}
if (xx.compare(half_pi) > 0)
{
eval_ldexp(half_pi, half_pi, 1);
eval_subtract(xx, half_pi, xx);
eval_ldexp(half_pi, half_pi, -1);
}
BOOST_ASSERT(xx.compare(half_pi) <= 0);
BOOST_ASSERT(xx.compare(ui_type(0)) >= 0);
}
else
{
n_pi = ui_type(1);
reduce_n_half_pi(xx, n_pi, true);
}

const bool b_zero = eval_get_sign(xx) == 0;

if (b_zero)
{
result = si_type(0);
}
else
{
eval_sin(result, xx);
}
if (b_negate_cos)
result.negate();
BOOST_MATH_INSTRUMENT_CODE(result.str(0, std::ios_base::scientific));
}

template <class T>
void eval_tan(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The tan function is only valid for floating point types.");
if (&result == &x)
{
T temp;
eval_tan(temp, x);
result = temp;
return;
}
T t;
eval_sin(result, x);
eval_cos(t, x);
eval_divide(result, t);
}

template <class T>
void hyp2F1(T& result, const T& a, const T& b, const T& c, const T& x)
{

typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;

T x_pow_n_div_n_fact(x);
T pochham_a(a);
T pochham_b(b);
T pochham_c(c);
T ap(a);
T bp(b);
T cp(c);

eval_multiply(result, pochham_a, pochham_b);
eval_divide(result, pochham_c);
eval_multiply(result, x_pow_n_div_n_fact);
eval_add(result, ui_type(1));

T lim;
eval_ldexp(lim, result, 1 - boost::multiprecision::detail::digits2<number<T, et_on> >::value());

if (eval_get_sign(lim) < 0)
lim.negate();

ui_type n;
T       term;

const unsigned series_limit =
boost::multiprecision::detail::digits2<number<T, et_on> >::value() < 100
? 100
: boost::multiprecision::detail::digits2<number<T, et_on> >::value();
for (n = 2; n < series_limit; ++n)
{
eval_multiply(x_pow_n_div_n_fact, x);
eval_divide(x_pow_n_div_n_fact, n);

eval_increment(ap);
eval_multiply(pochham_a, ap);
eval_increment(bp);
eval_multiply(pochham_b, bp);
eval_increment(cp);
eval_multiply(pochham_c, cp);

eval_multiply(term, pochham_a, pochham_b);
eval_divide(term, pochham_c);
eval_multiply(term, x_pow_n_div_n_fact);
eval_add(result, term);

if (eval_get_sign(term) < 0)
term.negate();
if (lim.compare(term) >= 0)
break;
}
if (n > series_limit)
BOOST_THROW_EXCEPTION(std::runtime_error("H2F1 failed to converge."));
}

template <class T>
void eval_asin(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The asin function is only valid for floating point types.");
typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;
typedef typename mpl::front<typename T::float_types>::type                          fp_type;

if (&result == &x)
{
T t(x);
eval_asin(result, t);
return;
}

switch (eval_fpclassify(x))
{
case FP_NAN:
case FP_INFINITE:
if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
BOOST_THROW_EXCEPTION(std::domain_error("Result is undefined or complex and there is no NaN for this number type."));
return;
case FP_ZERO:
result = x;
return;
default:;
}

const bool b_neg = eval_get_sign(x) < 0;

T xx(x);
if (b_neg)
xx.negate();

int c = xx.compare(ui_type(1));
if (c > 0)
{
if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
BOOST_THROW_EXCEPTION(std::domain_error("Result is undefined or complex and there is no NaN for this number type."));
return;
}
else if (c == 0)
{
result = get_constant_pi<T>();
eval_ldexp(result, result, -1);
if (b_neg)
result.negate();
return;
}

if (xx.compare(fp_type(1e-3)) < 0)
{
eval_multiply(xx, xx);
T t1, t2;
t1 = fp_type(0.5f);
t2 = fp_type(1.5f);
hyp2F1(result, t1, t1, t2, xx);
eval_multiply(result, x);
return;
}
else if (xx.compare(fp_type(1 - 5e-2f)) > 0)
{
T dx1;
T t1, t2;
eval_subtract(dx1, ui_type(1), xx);
t1 = fp_type(0.5f);
t2 = fp_type(1.5f);
eval_ldexp(dx1, dx1, -1);
hyp2F1(result, t1, t1, t2, dx1);
eval_ldexp(dx1, dx1, 2);
eval_sqrt(t1, dx1);
eval_multiply(result, t1);
eval_ldexp(t1, get_constant_pi<T>(), -1);
result.negate();
eval_add(result, t1);
if (b_neg)
result.negate();
return;
}
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
typedef typename boost::multiprecision::detail::canonical<long double, T>::type guess_type;
#else
typedef fp_type guess_type;
#endif
guess_type dd;
eval_convert_to(&dd, xx);

result = (guess_type)(std::asin(dd));


boost::intmax_t current_precision = eval_ilogb(result);
boost::intmax_t target_precision  = std::numeric_limits<number<T> >::is_specialized ? 
current_precision - 1 - (std::numeric_limits<number<T> >::digits * 2) / 3
: current_precision - 1 - (boost::multiprecision::detail::digits2<number<T> >::value() * 2) / 3;

while (current_precision > target_precision)
{
T sine, cosine;
eval_sin(sine, result);
eval_cos(cosine, result);
eval_subtract(sine, xx);
eval_divide(sine, cosine);
eval_subtract(result, sine);
current_precision = eval_ilogb(sine);
if (current_precision <= (std::numeric_limits<typename T::exponent_type>::min)() + 1)
break;
}
if (b_neg)
result.negate();
}

template <class T>
inline void eval_acos(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The acos function is only valid for floating point types.");
typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;

switch (eval_fpclassify(x))
{
case FP_NAN:
case FP_INFINITE:
if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
BOOST_THROW_EXCEPTION(std::domain_error("Result is undefined or complex and there is no NaN for this number type."));
return;
case FP_ZERO:
result = get_constant_pi<T>();
eval_ldexp(result, result, -1); 
return;
}

T xx;
eval_abs(xx, x);
int c = xx.compare(ui_type(1));

if (c > 0)
{
if (std::numeric_limits<number<T, et_on> >::has_quiet_NaN)
{
result = std::numeric_limits<number<T, et_on> >::quiet_NaN().backend();
errno  = EDOM;
}
else
BOOST_THROW_EXCEPTION(std::domain_error("Result is undefined or complex and there is no NaN for this number type."));
return;
}
else if (c == 0)
{
if (eval_get_sign(x) < 0)
result = get_constant_pi<T>();
else
result = ui_type(0);
return;
}

typedef typename mpl::front<typename T::float_types>::type fp_type;

if (xx.compare(fp_type(1e-3)) < 0)
{
eval_multiply(xx, xx);
T t1, t2;
t1 = fp_type(0.5f);
t2 = fp_type(1.5f);
hyp2F1(result, t1, t1, t2, xx);
eval_multiply(result, x);
eval_ldexp(t1, get_constant_pi<T>(), -1);
result.negate();
eval_add(result, t1);
return;
}
if (eval_get_sign(x) < 0)
{
eval_acos(result, xx);
result.negate();
eval_add(result, get_constant_pi<T>());
return;
}
else if (xx.compare(fp_type(0.85)) > 0)
{
T dx1;
T t1, t2;
eval_subtract(dx1, ui_type(1), xx);
t1 = fp_type(0.5f);
t2 = fp_type(1.5f);
eval_ldexp(dx1, dx1, -1);
hyp2F1(result, t1, t1, t2, dx1);
eval_ldexp(dx1, dx1, 2);
eval_sqrt(t1, dx1);
eval_multiply(result, t1);
return;
}

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
typedef typename boost::multiprecision::detail::canonical<long double, T>::type guess_type;
#else
typedef fp_type guess_type;
#endif
guess_type dd;
eval_convert_to(&dd, xx);

result = (guess_type)(std::acos(dd));


boost::intmax_t current_precision = eval_ilogb(result);
boost::intmax_t target_precision = std::numeric_limits<number<T> >::is_specialized ?
current_precision - 1 - (std::numeric_limits<number<T> >::digits * 2) / 3
: current_precision - 1 - (boost::multiprecision::detail::digits2<number<T> >::value() * 2) / 3;

while (current_precision > target_precision)
{
T sine, cosine;
eval_sin(sine, result);
eval_cos(cosine, result);
eval_subtract(cosine, xx);
cosine.negate();
eval_divide(cosine, sine);
eval_subtract(result, cosine);
current_precision = eval_ilogb(cosine);
if (current_precision <= (std::numeric_limits<typename T::exponent_type>::min)() + 1)
break;
}
}

template <class T>
void eval_atan(T& result, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The atan function is only valid for floating point types.");
typedef typename boost::multiprecision::detail::canonical<boost::int32_t, T>::type  si_type;
typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;
typedef typename mpl::front<typename T::float_types>::type                          fp_type;

switch (eval_fpclassify(x))
{
case FP_NAN:
result = x;
errno  = EDOM;
return;
case FP_ZERO:
result = x;
return;
case FP_INFINITE:
if (eval_get_sign(x) < 0)
{
eval_ldexp(result, get_constant_pi<T>(), -1);
result.negate();
}
else
eval_ldexp(result, get_constant_pi<T>(), -1);
return;
default:;
}

const bool b_neg = eval_get_sign(x) < 0;

T xx(x);
if (b_neg)
xx.negate();

if (xx.compare(fp_type(0.1)) < 0)
{
T t1, t2, t3;
t1 = ui_type(1);
t2 = fp_type(0.5f);
t3 = fp_type(1.5f);
eval_multiply(xx, xx);
xx.negate();
hyp2F1(result, t1, t2, t3, xx);
eval_multiply(result, x);
return;
}

if (xx.compare(fp_type(10)) > 0)
{
T t1, t2, t3;
t1 = fp_type(0.5f);
t2 = ui_type(1u);
t3 = fp_type(1.5f);
eval_multiply(xx, xx);
eval_divide(xx, si_type(-1), xx);
hyp2F1(result, t1, t2, t3, xx);
eval_divide(result, x);
if (!b_neg)
result.negate();
eval_ldexp(t1, get_constant_pi<T>(), -1);
eval_add(result, t1);
if (b_neg)
result.negate();
return;
}

fp_type d;
eval_convert_to(&d, xx);
result = fp_type(std::atan(d));


boost::intmax_t current_precision = eval_ilogb(result);
boost::intmax_t target_precision  = std::numeric_limits<number<T> >::is_specialized ?
current_precision - 1 - (std::numeric_limits<number<T> >::digits * 2) / 3
: current_precision - 1 - (boost::multiprecision::detail::digits2<number<T> >::value() * 2) / 3;

T s, c, t;
while (current_precision > target_precision)
{
eval_sin(s, result);
eval_cos(c, result);
eval_multiply(t, xx, c);
eval_subtract(t, s);
eval_multiply(s, t, c);
eval_add(result, s);
current_precision = eval_ilogb(s);
if (current_precision <= (std::numeric_limits<typename T::exponent_type>::min)() + 1)
break;
}
if (b_neg)
result.negate();
}

template <class T>
void eval_atan2(T& result, const T& y, const T& x)
{
BOOST_STATIC_ASSERT_MSG(number_category<T>::value == number_kind_floating_point, "The atan2 function is only valid for floating point types.");
if (&result == &y)
{
T temp(y);
eval_atan2(result, temp, x);
return;
}
else if (&result == &x)
{
T temp(x);
eval_atan2(result, y, temp);
return;
}

typedef typename boost::multiprecision::detail::canonical<boost::uint32_t, T>::type ui_type;

switch (eval_fpclassify(y))
{
case FP_NAN:
result = y;
errno  = EDOM;
return;
case FP_ZERO:
{
if (eval_signbit(x))
{
result = get_constant_pi<T>();
if (eval_signbit(y))
result.negate();
}
else
{
result = y; 
}
return;
}
case FP_INFINITE:
{
if (eval_fpclassify(x) == FP_INFINITE)
{
if (eval_signbit(x))
{
eval_ldexp(result, get_constant_pi<T>(), -2);
eval_subtract(result, get_constant_pi<T>());
if (eval_get_sign(y) >= 0)
result.negate();
}
else
{
eval_ldexp(result, get_constant_pi<T>(), -2);
if (eval_get_sign(y) < 0)
result.negate();
}
}
else
{
eval_ldexp(result, get_constant_pi<T>(), -1);
if (eval_get_sign(y) < 0)
result.negate();
}
return;
}
}

switch (eval_fpclassify(x))
{
case FP_NAN:
result = x;
errno  = EDOM;
return;
case FP_ZERO:
{
eval_ldexp(result, get_constant_pi<T>(), -1);
if (eval_get_sign(y) < 0)
result.negate();
return;
}
case FP_INFINITE:
if (eval_get_sign(x) > 0)
result = ui_type(0);
else
result = get_constant_pi<T>();
if (eval_get_sign(y) < 0)
result.negate();
return;
}

T xx;
eval_divide(xx, y, x);
if (eval_get_sign(xx) < 0)
xx.negate();

eval_atan(result, xx);

const bool y_neg = eval_get_sign(y) < 0;
const bool x_neg = eval_get_sign(x) < 0;

if (y_neg != x_neg)
result.negate();

if (x_neg)
{
if (y_neg)
eval_subtract(result, get_constant_pi<T>());
else
eval_add(result, get_constant_pi<T>());
}
}
template <class T, class A>
inline typename enable_if<is_arithmetic<A>, void>::type eval_atan2(T& result, const T& x, const A& a)
{
typedef typename boost::multiprecision::detail::canonical<A, T>::type          canonical_type;
typedef typename mpl::if_<is_same<A, canonical_type>, T, canonical_type>::type cast_type;
cast_type                                                                      c;
c = a;
eval_atan2(result, x, c);
}

template <class T, class A>
inline typename enable_if<is_arithmetic<A>, void>::type eval_atan2(T& result, const A& x, const T& a)
{
typedef typename boost::multiprecision::detail::canonical<A, T>::type          canonical_type;
typedef typename mpl::if_<is_same<A, canonical_type>, T, canonical_type>::type cast_type;
cast_type                                                                      c;
c = x;
eval_atan2(result, c, a);
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
