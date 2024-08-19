

#ifndef BOOST_MATH_SF_LAMBERT_W_HPP
#define BOOST_MATH_SF_LAMBERT_W_HPP

#ifdef _MSC_VER
#pragma warning(disable : 4127)
#endif





#include <boost/math/policies/error_handling.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/log1p.hpp> 
#include <boost/math/constants/constants.hpp> 
#include <boost/math/special_functions/pow.hpp> 
#include <boost/math/tools/series.hpp> 
#include <boost/math/tools/rational.hpp>  
#include <boost/mpl/int.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/math/tools/precision.hpp> 
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/tools/cxx03_warn.hpp>

#include <limits>
#include <cmath>
#include <limits>
#include <exception>

#include <iostream>
#include <typeinfo>
#include <boost/math/special_functions/next.hpp>  

typedef double lookup_t; 

#include <boost/math/special_functions/detail/lambert_w_lookup_table.ipp>

#if defined(__GNUC__) && defined(BOOST_MATH_USE_FLOAT128)
#pragma GCC system_header
#endif

namespace boost {
namespace math {
namespace lambert_w_detail {


template <class T>
inline T lambert_w_halley_step(T w_est, const T z)
{
BOOST_MATH_STD_USING
T e = exp(w_est);
w_est -= 2 * (w_est + 1) * (e * w_est - z) / (z * (w_est + 2) + e * (w_est * (w_est + 2) + 2));
return w_est;
} 


template <class T>
inline
T lambert_w_halley_iterate(T w_est, const T z)
{
BOOST_MATH_STD_USING
static const T max_diff = boost::math::tools::root_epsilon<T>() * fabs(w_est);

T w_new = lambert_w_halley_step(w_est, z);
T diff = fabs(w_est - w_new);
while (diff > max_diff)
{
w_est = w_new;
w_new = lambert_w_halley_step(w_est, z);
diff = fabs(w_est - w_new);
}
return w_new;
} 

template <class T>
inline
T lambert_w_maybe_halley_iterate(T z, T w, boost::false_type const&)
{
return lambert_w_halley_step(z, w); 
}

template <class T>
inline
T lambert_w_maybe_halley_iterate(T z, T w, boost::true_type const&)
{
return lambert_w_halley_iterate(z, w); 
}


template <class T>
inline
double maybe_reduce_to_double(const T& z, const boost::true_type&)
{
return static_cast<double>(z); 
}

template <class T>
inline
T maybe_reduce_to_double(const T& z, const boost::false_type&)
{ 
return z;
}

template <class T>
inline
double must_reduce_to_double(const T& z, const boost::true_type&)
{
return static_cast<double>(z); 
}

template <class T>
inline
double must_reduce_to_double(const T& z, const boost::false_type&)
{ 
return boost::lexical_cast<double>(z);
}


template<typename T>
inline
T schroeder_update(const T w, const T y)
{

BOOST_MATH_STD_USING 
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SCHROEDER
std::streamsize saved_precision = std::cout.precision(std::numeric_limits<T>::max_digits10);
using boost::math::float_distance;
T fd = float_distance<T>(w, y);
std::cout << "Schroder ";
if (abs(fd) < 214748000.)
{
std::cout << " Distance = "<< static_cast<int>(fd);
}
else
{
std::cout << "Difference w - y = " << (w - y) << ".";
}
std::cout << std::endl;
#endif 
const T f0 = w - y; 
const T f1 = 1 + y; 
const T f00 = f0 * f0;
const T f11 = f1 * f1;
const T f0y = f0 * y;
const T result =
w - 4 * f0 * (6 * f1 * (f11 + f0y)  +  f00 * y) /
(f11 * (24 * f11 + 36 * f0y) +
f00 * (6 * y * y  +  8 * f1 * y  +  f0y)); 

#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SCHROEDER
std::cout << "Schroeder refined " << w << "  " << y << ", difference  " << w-y  << ", change " << w - result << ", to result " << result << std::endl;
std::cout.precision(saved_precision); 
#endif 

return result;
} 


template<typename T>
T lambert_w_singularity_series(const T p)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SINGULARITY_SERIES
std::size_t saved_precision = std::cout.precision(3);
std::cout << "Singularity_series Lambert_w p argument = " << p << std::endl;
std::cout
<< std::endl;
std::cout.precision(saved_precision);
#endif 

static const T q[] =
{
-static_cast<T>(1), 
+T(1), 
-T(1) / 3, 
+T(11) / 72, 
-T(43) / 540, 
+T(769) / 17280, 
-T(221) / 8505, 
+T(680863uLL) / 43545600uLL, 
-T(1963uLL) / 204120uLL, 
+T(226287557uLL) / 37623398400uLL, 
-T(5776369uLL) / 1515591000uLL, 
+T(169709463197uLL) / 69528040243200uLL, 
-T(1118511313uLL) / 709296588000uLL, 
+T(667874164916771uLL) / 650782456676352000uLL, 
-T(500525573uLL) / 744761417400uLL, 
BOOST_MATH_BIG_CONSTANT(T, 64, +0.000442473061814620910), 
BOOST_MATH_BIG_CONSTANT(T, 64, -0.000292677224729627445), 
BOOST_MATH_BIG_CONSTANT(T, 64, 0.000194387276054539318), 
BOOST_MATH_BIG_CONSTANT(T, 64, -0.000129574266852748819), 
BOOST_MATH_BIG_CONSTANT(T, 64, +0.0000866503580520812717), 
BOOST_MATH_BIG_CONSTANT(T, 113, -0.000058113607504413816772205464778828177256611844221913) 
}; 




BOOST_MATH_STD_USING 

const T absp = abs(p);

#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_TERMS
{
int terms = 20; 
if (absp < 0.01159)
{ 
terms = 6;
}
else if (absp < 0.0766)
{ 
terms = 10;
}
std::streamsize saved_precision = std::cout.precision(3);
std::cout << "abs(p) = " << absp << ", terms = " << terms << std::endl;
std::cout.precision(saved_precision);
}
#endif 

if (absp < 0.01159)
{ 
return
-1 +
p * (1 +
p * (q[2] +
p * (q[3] +
p * (q[4] +
p * (q[5] +
p * q[6]
)))));
}
else if (absp < 0.0766) 
{ 
return
-1 +
p * (1 +
p * (q[2] +
p * (q[3] +
p * (q[4] +
p * (q[5] +
p * (q[6] +
p * (q[7] +
p * (q[8] +
p * (q[9] +
p * q[10]
)))))))));
}
else
{ 
return
-1 +
p * (1 +
p * (q[2] +
p * (q[3] +
p * (q[4] +
p * (q[5] +
p * (q[6] +
p * (q[7] +
p * (q[8] +
p * (q[9] +
p * (q[10] +
p * (q[11] +
p * (q[12] +
p * (q[13] +
p * (q[14] +
p * (q[15] +
p * (q[16] +
p * (q[17] +
p * (q[18] +
p * (q[19] +
p * q[20] 
)))))))))))))))))));
}
} 











template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy&, boost::integral_constant<int, 0> const&);   

template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy&, boost::integral_constant<int, 1> const&);   

template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy&, boost::integral_constant<int, 2> const&);   

template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy&, boost::integral_constant<int, 3> const&);   

template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy&, boost::integral_constant<int, 4> const&);   

template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy&, boost::integral_constant<int, 5> const&);   
template <class T, class Policy>
T lambert_w0_small_z(T x, const Policy& pol)
{ 
typedef boost::integral_constant<int,
std::numeric_limits<T>::is_specialized == 0 ? 5 :
#ifndef BOOST_NO_CXX11_NUMERIC_LIMITS
std::numeric_limits<T>::max_digits10 <=  9 ? 0 : 
std::numeric_limits<T>::max_digits10 <= 17 ? 1 : 
std::numeric_limits<T>::max_digits10 <= 22 ? 2 : 
std::numeric_limits<T>::max_digits10 <  37 ? 4  
#else
std::numeric_limits<T>::radix != 2 ? 5 :
std::numeric_limits<T>::digits <= 24 ? 0 : 
std::numeric_limits<T>::digits <= 53 ? 1 : 
std::numeric_limits<T>::digits <= 64 ? 2 : 
std::numeric_limits<T>::digits <= 113 ? 4  
#endif
:  5                                            
> tag_type;
return lambert_w0_small_z(x, pol, tag_type());
} 


template <class T, class Policy>
T lambert_w0_small_z(T z, const Policy&, boost::integral_constant<int, 0> const&)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::streamsize prec = std::cout.precision(std::numeric_limits<T>::max_digits10); 
std::cout << "\ntag_type 0 float lambert_w0_small_z called with z = " << z << " using " << 9 << " terms of precision "
<< std::numeric_limits<float>::max_digits10 << " decimal digits. " << std::endl;
#endif 
T result =
z * (1 - 
z * (1 -  
z * (static_cast<float>(3uLL) / 2uLL - 
z * (2.6666666666666666667F -  
z * (5.2083333333333333333F - 
z * (10.8F - 
z * (23.343055555555555556F - 
z * (52.012698412698412698F - 
z * 118.62522321428571429F)))))))); 

#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::cout << "return w = " << result << std::endl;
std::cout.precision(prec); 
#endif 

return result;
} 


template <class T, class Policy>
T lambert_w0_small_z(const T z, const Policy&, boost::integral_constant<int, 1> const&)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::streamsize prec = std::cout.precision(std::numeric_limits<T>::max_digits10); 
std::cout << "\ntag_type 1 double lambert_w0_small_z called with z = " << z << " using " << 17 << " terms of precision, "
<< std::numeric_limits<double>::max_digits10 << " decimal digits. " << std::endl;
#endif 
T result =
z * (1. - 
z * (1. -  
z * (1.5 - 
z * (2.6666666666666666667 -  
z * (5.2083333333333333333 - 
z * (10.8 - 
z * (23.343055555555555556 - 
z * (52.012698412698412698 - 
z * (118.62522321428571429 - 
z * (275.57319223985890653 - 
z * (649.78717234347442681 - 
z * (1551.1605194805194805 - 
z * (3741.4497029592385495 - 
z * (9104.5002411580189358 - 
z * (22324.308512706601434 - 
z * (55103.621972903835338 - 
z * 136808.86090394293563)))))))))))))))); 

#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::cout << "return w = " << result << std::endl;
std::cout.precision(prec); 
#endif 

return result;
} 

template <class T, class Policy>
T lambert_w0_small_z(const T z, const Policy&, boost::integral_constant<int, 2> const&)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::streamsize precision = std::cout.precision(std::numeric_limits<T>::max_digits10); 
std::cout << "\ntag_type 2 long double (80-bit double extended) lambert_w0_small_z called with z = " << z << " using " << 21 << " terms of precision, "
<< std::numeric_limits<long double>::max_digits10 << " decimal digits. " << std::endl;
#endif 

T result =
z * (1.L - 
z * (1.L - 
z * (1.500000000000000000000000000000000L - 
z * (2.666666666666666666666666666666666L - 
z * (5.208333333333333333333333333333333L - 
z * (10.80000000000000000000000000000000L - 
z * (23.34305555555555555555555555555555L - 
z * (52.01269841269841269841269841269841L - 
z * (118.6252232142857142857142857142857L - 
z * (275.5731922398589065255731922398589L - 
z * (649.7871723434744268077601410934744L - 
z * (1551.160519480519480519480519480519L - 
z * (3741.449702959238549516327294105071L - 
z * (9104.500241158018935796713574491352L - 
z * (22324.308512706601434280005708577137L - 
z * (55103.621972903835337697771560205422L - 
z * (136808.86090394293563342215789305736L - 
z * (341422.05066583836331735491399356945L - 
z * (855992.9659966075514633630250633224L - 
z * (2.154990206091088289321708745358647e6L 
))))))))))))))))))));

#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::cout << "return w = " << result << std::endl;
std::cout.precision(precision); 
#endif 
return result;
}  


template <class T, class Policy>
T lambert_w0_small_z(const T z, const Policy&, boost::integral_constant<int, 3> const&)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::streamsize precision = std::cout.precision(std::numeric_limits<T>::max_digits10); 
std::cout << "\ntag_type 3 long double (128-bit) lambert_w0_small_z called with z = " << z << " using " << 17 << " terms of precision,  "
<< std::numeric_limits<double>::max_digits10 << " decimal digits. " << std::endl;
#endif 
T  result =
z * (1.L - 
z * (1.L -  
z * (1.5L - 
z * (2.6666666666666666666666666666666666L -  
z * (5.2052083333333333333333333333333333L - 
z * (10.800000000000000000000000000000000L - 
z * (23.343055555555555555555555555555555L - 
z * (52.0126984126984126984126984126984126L - 
z * (118.625223214285714285714285714285714L - 
z * (275.57319223985890652557319223985890L - 
z * (649.78717234347442680776014109347442680776014109347L - 
z * (1551.1605194805194805194805194805194805194805194805L - 
z * (3741.4497029592385495163272941050718828496606274384L - 
z * (9104.5002411580189357967135744913522691300469078247L - 
z * (22324.308512706601434280005708577137148565719994291L - 
z * (55103.621972903835337697771560205422639285073147507L - 
z * 136808.86090394293563342215789305736395683485630576L    
))))))))))))))));

#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::cout << "return w = " << result << std::endl;
std::cout.precision(precision); 
#endif 
return result;
}  


#ifdef BOOST_HAS_FLOAT128
template <class T, class Policy>
T lambert_w0_small_z(const T z, const Policy&, boost::integral_constant<int, 4> const&)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::streamsize precision = std::cout.precision(std::numeric_limits<T>::max_digits10); 
std::cout << "\ntag_type 4 128-bit quad float128 lambert_w0_small_z called with z = " << z << " using " << 34 << " terms of precision, "
<< std::numeric_limits<float128>::max_digits10 << " max decimal digits." << std::endl;
#endif 
T  result =
z * (1.Q - 
z * (1.Q - 
z * (1.500000000000000000000000000000000Q - 
z * (2.666666666666666666666666666666666Q - 
z * (5.208333333333333333333333333333333Q - 
z * (10.80000000000000000000000000000000Q - 
z * (23.34305555555555555555555555555555Q - 
z * (52.01269841269841269841269841269841Q - 
z * (118.6252232142857142857142857142857Q - 
z * (275.5731922398589065255731922398589Q - 
z * (649.7871723434744268077601410934744Q - 
z * (1551.160519480519480519480519480519Q - 
z * (3741.449702959238549516327294105071Q - 
z * (9104.500241158018935796713574491352Q - 
z * (22324.308512706601434280005708577137Q - 
z * (55103.621972903835337697771560205422Q - 
z * (136808.86090394293563342215789305736Q - 
z * (341422.05066583836331735491399356945Q - 
z * (855992.9659966075514633630250633224Q - 
z * (2.154990206091088289321708745358647e6Q - 
z * (5.445552922314462431642316420035073e6Q - 
z * (1.380733000216662949061923813184508e7Q - 
z * (3.511704498513923292853869855945334e7Q - 
z * (8.956800256102797693072819557780090e7Q - 
z * (2.290416846187949813964782641734774e8Q - 
z * (5.871035041171798492020292225245235e8Q - 
z * (1.508256053857792919641317138812957e9Q - 
z * (3.882630161293188940385873468413841e9Q - 
z * (1.001394313665482968013913601565723e10Q - 
z * (2.587356736265760638992878359024929e10Q - 
z * (6.696209709358073856946120522333454e10Q - 
z * (1.735711659599198077777078238043644e11Q - 
z * (4.505680465642353886756098108484670e11Q - 
z * (1.171223178256487391904047636564823e12Q  
))))))))))))))))))))))))))))))))));


#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::cout << "return w = " << result << std::endl;
std::cout.precision(precision); 
#endif 

return result;
}  

#else

template <class T, class Policy>
inline T lambert_w0_small_z(const T z, const Policy& pol, boost::integral_constant<int, 4> const&)
{
return lambert_w0_small_z(z, pol, boost::integral_constant<int, 5>());
}

#endif 

template <class T>
struct lambert_w0_small_z_series_term
{
typedef T result_type;


lambert_w0_small_z_series_term(T _z, T _term, int _k)
: k(_k), z(_z), term(_term) { }

T operator()()
{ 
using std::pow;
++k;
term *= -z / k;
T result = term * pow(T(k), -1 + k); 
return result; 
}
private:
int k;
T z;
T term;
}; 

template <class T, class Policy>
inline T lambert_w0_small_z(T z, const Policy& pol, boost::integral_constant<int, 5> const&)
{
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::streamsize precision = std::cout.precision(std::numeric_limits<T>::max_digits10); 
std::cout << "Generic lambert_w0_small_z called with z = " << z << " using as many terms needed for precision." << std::endl;
std::cout << "Argument z is of type " << typeid(T).name() << std::endl;
#endif 


static const T coeff[] =
{
0, 
1, 
-1, 
static_cast<T>(3uLL) / 2uLL, 
-static_cast<T>(8uLL) / 3uLL, 
static_cast<T>(125uLL) / 24uLL, 
-static_cast<T>(54uLL) / 5uLL, 
static_cast<T>(16807uLL) / 720uLL, 
-static_cast<T>(16384uLL) / 315uLL, 
static_cast<T>(531441uLL) / 4480uLL, 
-static_cast<T>(156250uLL) / 567uLL, 
static_cast<T>(2357947691uLL) / 3628800uLL, 
-static_cast<T>(2985984uLL) / 1925uLL, 
static_cast<T>(1792160394037uLL) / 479001600uLL, 
-static_cast<T>(7909306972uLL) / 868725uLL, 
static_cast<T>(320361328125uLL) / 14350336uLL, 
-static_cast<T>(35184372088832uLL) / 638512875uLL, 
static_cast<T>(2862423051509815793uLL) / 20922789888000uLL, 
-static_cast<T>(5083731656658uLL) / 14889875uLL,


}; 



using boost::math::policies::get_epsilon; 
using boost::math::tools::sum_series;
using boost::math::tools::evaluate_polynomial;


T result = evaluate_polynomial(coeff, z);


lambert_w0_small_z_series_term<T> s(z, -pow<18>(z) / 6402373705728000uLL, 18);


boost::uintmax_t max_iter = policies::get_max_series_iterations<Policy>(); 
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES
std::cout << "max iter from policy = " << max_iter << std::endl;
#endif 

result = sum_series(s, get_epsilon<T, Policy>(), max_iter, result);


policies::check_series_iterations<T>("boost::math::lambert_w0_small_z<%1%>(%1%)", max_iter, pol);
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_W_SMALL_Z_SERIES_ITERATIONS
std::cout << "z = " << z << " needed  " << max_iter << " iterations." << std::endl;
std::cout.precision(prec); 
#endif 
return result;
} 

template <typename T>
inline
T lambert_w0_approx(T z)
{
BOOST_MATH_STD_USING
T lz = log(z);
T llz = log(lz);
T w = lz - llz + (llz / lz); 
return w;
}





template <class T>
inline T do_get_near_singularity_param(T z)
{
BOOST_MATH_STD_USING
const T p2 = 2 * (boost::math::constants::e<T>() * z + 1);
const T p = sqrt(p2);
return p;
}
template <class T, class Policy>
inline T get_near_singularity_param(T z, const Policy)
{
typedef typename policies::evaluation<T, Policy>::type value_type;
return static_cast<T>(do_get_near_singularity_param(static_cast<value_type>(z)));
}



template <class T>
T lambert_w_positive_rational_float(T z)
{
BOOST_MATH_STD_USING
if (z < 2)
{
if (z < 0.5)
{ 
static const T Y = 8.196592331e-01f;
static const T P[] = {
1.803388345e-01f,
-4.820256838e-01f,
-1.068349741e+00f,
-3.506624319e-02f,
};
static const T Q[] = {
1.000000000e+00f,
2.871703469e+00f,
1.690949264e+00f,
};
return z * (Y + boost::math::tools::evaluate_polynomial(P, z) / boost::math::tools::evaluate_polynomial(Q, z));
}
else
{ 
static const T Y = 5.503368378e-01f;
static const T P[] = {
4.493332766e-01f,
2.543432707e-01f,
-4.808788799e-01f,
-1.244425316e-01f,
};
static const T Q[] = {
1.000000000e+00f,
2.780661241e+00f,
1.830840318e+00f,
2.407221031e-01f,
};
return z * (Y + boost::math::tools::evaluate_rational(P, Q, z));
}
}
else if (z < 6)
{
static const T Y = 1.162393570e+00f;
static const T P[] = {
-1.144183394e+00f,
-4.712732855e-01f,
1.563162512e-01f,
1.434010911e-02f,
};
static const T Q[] = {
1.000000000e+00f,
1.192626340e+00f,
2.295580708e-01f,
5.477869455e-03f,
};
return Y + boost::math::tools::evaluate_rational(P, Q, z);
}
else if (z < 18)
{
static const T Y = 1.809371948e+00f;
static const T P[] = {
-1.689291769e+00f,
-3.337812742e-01f,
3.151434873e-02f,
1.134178734e-03f,
};
static const T Q[] = {
1.000000000e+00f,
5.716915685e-01f,
4.489521292e-02f,
4.076716763e-04f,
};
return Y + boost::math::tools::evaluate_rational(P, Q, z);
}
else if (z < 9897.12905874)  
{
static const T Y = -1.402973175e+00f;
static const T P[] = {
1.966174312e+00f,
2.350864728e-01f,
-5.098074353e-02f,
-1.054818339e-02f,
};
static const T Q[] = {
1.000000000e+00f,
4.388208264e-01f,
8.316639634e-02f,
3.397187918e-03f,
-1.321489743e-05f,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_polynomial(P, log_w) / boost::math::tools::evaluate_polynomial(Q, log_w);
}
else if (z < 7.896296e+13)  
{
static const T Y = -2.735729218e+00f;
static const T P[] = {
3.424903470e+00f,
7.525631787e-02f,
-1.427309584e-02f,
-1.435974178e-05f,
};
static const T Q[] = {
1.000000000e+00f,
2.514005579e-01f,
6.118994652e-03f,
-1.357889535e-05f,
7.312865624e-08f,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_polynomial(P, log_w) / boost::math::tools::evaluate_polynomial(Q, log_w);
}
else 
{
static const T Y = -4.012863159e+00f;
static const T P[] = {
4.431629226e+00f,
2.756690487e-01f,
-2.992956930e-03f,
-4.912259384e-05f,
};
static const T Q[] = {
1.000000000e+00f,
2.015434591e-01f,
4.949426142e-03f,
1.609659944e-05f,
-5.111523436e-09f,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_polynomial(P, log_w) / boost::math::tools::evaluate_polynomial(Q, log_w);
}
}

template <class T, class Policy>
T lambert_w_negative_rational_float(T z, const Policy& pol)
{
BOOST_MATH_STD_USING
if (z > -0.27)
{
if (z < -0.051)
{
static const T Y = 1.255809784e+00f;
static const T P[] = {
-2.558083412e-01f,
-2.306524098e+00f,
-5.630887033e+00f,
-3.803974556e+00f,
};
static const T Q[] = {
1.000000000e+00f,
5.107680783e+00f,
7.914062868e+00f,
3.501498501e+00f,
};
return z * (Y + boost::math::tools::evaluate_rational(P, Q, z));
}
else
{
return lambert_w0_small_z(z, pol);
}
}
else if (z > -0.3578794411714423215955237701)
{ 
static const T Y = 1.220928431e-01f;
static const T P[] = {
-1.221787446e-01f,
-6.816155875e+00f,
7.144582035e+01f,
1.128444390e+03f,
};
static const T Q[] = {
1.000000000e+00f,
6.480326790e+01f,
1.869145243e+02f,
-1.361804274e+03f,
1.117826726e+03f,
};
T d = z + 0.367879441171442321595523770161460867445811f;
return -d / (Y + boost::math::tools::evaluate_polynomial(P, d) / boost::math::tools::evaluate_polynomial(Q, d));
}
else
{
return lambert_w_singularity_series(get_near_singularity_param(z, pol));
}
}

template <class T, class Policy>
inline T lambert_w0_imp(T z, const Policy& pol, const boost::integral_constant<int, 1>&)
{
static const char* function = "boost::math::lambert_w0<%1%>"; 
BOOST_MATH_STD_USING 

if ((boost::math::isnan)(z))
{
return boost::math::policies::raise_domain_error<T>(function, "Expected a value > -e^-1 (-0.367879...) but got %1%.", z, pol);
}
if ((boost::math::isinf)(z))
{
return boost::math::policies::raise_overflow_error<T>(function, "Expected a finite value but got %1%.", z, pol);
}

if (z >= 0.05) 
{ 
return lambert_w_positive_rational_float(z);
}
else if (z <= -0.3678794411714423215955237701614608674458111310f)
{
if (z < -0.3678794411714423215955237701614608674458111310f)
return boost::math::policies::raise_domain_error<T>(function, "Expected z >= -e^-1 (-0.367879...) but got %1%.", z, pol);
return -1;
}
else 
{
return lambert_w_negative_rational_float(z, pol);
}
} 

template <class T>
T lambert_w_positive_rational_double(T z)
{
BOOST_MATH_STD_USING
if (z < 2)
{
if (z < 0.5)
{
static const T offset = 8.19659233093261719e-01;
static const T P[] = {
1.80340766906685177e-01,
3.28178241493119307e-01,
-2.19153620687139706e+00,
-7.24750929074563990e+00,
-7.28395876262524204e+00,
-2.57417169492512916e+00,
-2.31606948888704503e-01
};
static const T Q[] = {
1.00000000000000000e+00,
7.36482529307436604e+00,
2.03686007856430677e+01,
2.62864592096657307e+01,
1.59742041380858333e+01,
4.03760534788374589e+00,
2.91327346750475362e-01
};
return z * (offset + boost::math::tools::evaluate_polynomial(P, z) / boost::math::tools::evaluate_polynomial(Q, z));
}
else
{
static const T offset = 5.50335884094238281e-01;
static const T P[] = {
4.49664083944098322e-01,
1.90417666196776909e+00,
1.99951368798255994e+00,
-6.91217310299270265e-01,
-1.88533935998617058e+00,
-7.96743968047750836e-01,
-1.02891726031055254e-01,
-3.09156013592636568e-03
};
static const T Q[] = {
1.00000000000000000e+00,
6.45854489419584014e+00,
1.54739232422116048e+01,
1.72606164253337843e+01,
9.29427055609544096e+00,
2.29040824649748117e+00,
2.21610620995418981e-01,
5.70597669908194213e-03
};
return z * (offset + boost::math::tools::evaluate_rational(P, Q, z));
}
}
else if (z < 6)
{
static const T Y = 1.16239356994628906e+00;
static const T P[] = {
-1.16230494982099475e+00,
-3.38528144432561136e+00,
-2.55653717293161565e+00,
-3.06755172989214189e-01,
1.73149743765268289e-01,
3.76906042860014206e-02,
1.84552217624706666e-03,
1.69434126904822116e-05,
};
static const T Q[] = {
1.00000000000000000e+00,
3.77187616711220819e+00,
4.58799960260143701e+00,
2.24101228462292447e+00,
4.54794195426212385e-01,
3.60761772095963982e-02,
9.25176499518388571e-04,
4.43611344705509378e-06,
};
return Y + boost::math::tools::evaluate_rational(P, Q, z);
}
else if (z < 18)
{
static const T offset = 1.80937194824218750e+00;
static const T P[] =
{
-1.80690935424793635e+00,
-3.66995929380314602e+00,
-1.93842957940149781e+00,
-2.94269984375794040e-01,
1.81224710627677778e-03,
2.48166798603547447e-03,
1.15806592415397245e-04,
1.43105573216815533e-06,
3.47281483428369604e-09
};
static const T Q[] = {
1.00000000000000000e+00,
2.57319080723908597e+00,
1.96724528442680658e+00,
5.84501352882650722e-01,
7.37152837939206240e-02,
3.97368430940416778e-03,
8.54941838187085088e-05,
6.05713225608426678e-07,
8.17517283816615732e-10
};
return offset + boost::math::tools::evaluate_rational(P, Q, z);
}
else if (z < 9897.12905874)  
{
static const T Y = -1.40297317504882812e+00;
static const T P[] = {
1.97011826279311924e+00,
1.05639945701546704e+00,
3.33434529073196304e-01,
3.34619153200386816e-02,
-5.36238353781326675e-03,
-2.43901294871308604e-03,
-2.13762095619085404e-04,
-4.85531936495542274e-06,
-2.02473518491905386e-08,
};
static const T Q[] = {
1.00000000000000000e+00,
8.60107275833921618e-01,
4.10420467985504373e-01,
1.18444884081994841e-01,
2.16966505556021046e-02,
2.24529766630769097e-03,
9.82045090226437614e-05,
1.36363515125489502e-06,
3.44200749053237945e-09,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_rational(P, Q, log_w);
}
else if (z < 7.896296e+13)  
{
static const T Y = -2.73572921752929688e+00;
static const T P[] = {
3.30547638424076217e+00,
1.64050071277550167e+00,
4.57149576470736039e-01,
4.03821227745424840e-02,
-4.99664976882514362e-04,
-1.28527893803052956e-04,
-2.95470325373338738e-06,
-1.76662025550202762e-08,
-1.98721972463709290e-11,
};
static const T Q[] = {
1.00000000000000000e+00,
6.91472559412458759e-01,
2.48154578891676774e-01,
4.60893578284335263e-02,
3.60207838982301946e-03,
1.13001153242430471e-04,
1.33690948263488455e-06,
4.97253225968548872e-09,
3.39460723731970550e-12,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_rational(P, Q, log_w);
}
else if (z < 2.6881171e+43) 
{
static const T Y = -4.01286315917968750e+00;
static const T P[] = {
5.07714858354309672e+00,
-3.32994414518701458e+00,
-8.61170416909864451e-01,
-4.01139705309486142e-02,
-1.85374201771834585e-04,
1.08824145844270666e-05,
1.17216905810452396e-07,
2.97998248101385990e-10,
1.42294856434176682e-13,
};
static const T Q[] = {
1.00000000000000000e+00,
-4.85840770639861485e-01,
-3.18714850604827580e-01,
-3.20966129264610534e-02,
-1.06276178044267895e-03,
-1.33597828642644955e-05,
-6.27900905346219472e-08,
-9.35271498075378319e-11,
-2.60648331090076845e-14,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_rational(P, Q, log_w);
}
else 
{
static const T Y = -5.70115661621093750e+00;
static const T P[] = {
6.42275660145116698e+00,
1.33047964073367945e+00,
6.72008923401652816e-02,
1.16444069958125895e-03,
7.06966760237470501e-06,
5.48974896149039165e-09,
-7.00379652018853621e-11,
-1.89247635913659556e-13,
-1.55898770790170598e-16,
-4.06109208815303157e-20,
-2.21552699006496737e-24,
};
static const T Q[] = {
1.00000000000000000e+00,
3.34498588416632854e-01,
2.51519862456384983e-02,
6.81223810622416254e-04,
7.94450897106903537e-06,
4.30675039872881342e-08,
1.10667669458467617e-10,
1.31012240694192289e-13,
6.53282047177727125e-17,
1.11775518708172009e-20,
3.78250395617836059e-25,
};
T log_w = log(z);
return log_w + Y + boost::math::tools::evaluate_rational(P, Q, log_w);
}
}

template <class T, class Policy>
T lambert_w_negative_rational_double(T z, const Policy& pol)
{
BOOST_MATH_STD_USING
if (z > -0.1)
{
if (z < -0.051)
{
static const T Y = 1.08633995056152344e+00;
static const T P[] = {
-8.63399505615014331e-02,
-1.64303871814816464e+00,
-7.71247913918273738e+00,
-1.41014495545382454e+01,
-1.02269079949257616e+01,
-2.17236002836306691e+00,
};
static const T Q[] = {
1.00000000000000000e+00,
7.44775406945739243e+00,
2.04392643087266541e+01,
2.51001961077774193e+01,
1.31256080849023319e+01,
2.11640324843601588e+00,
};
return z * (Y + boost::math::tools::evaluate_rational(P, Q, z));
}
else
{
return lambert_w0_small_z(z, pol);
}
}
else if (z > -0.2)
{
static const T Y = 1.20359611511230469e+00;
static const T P[] = {
-2.03596115108465635e-01,
-2.95029082937201859e+00,
-1.54287922188671648e+01,
-3.81185809571116965e+01,
-4.66384358235575985e+01,
-2.59282069989642468e+01,
-4.70140451266553279e+00,
};
static const T Q[] = {
1.00000000000000000e+00,
9.57921436074599929e+00,
3.60988119290234377e+01,
6.73977699505546007e+01,
6.41104992068148823e+01,
2.82060127225153607e+01,
4.10677610657724330e+00,
};
return z * (Y + boost::math::tools::evaluate_rational(P, Q, z));
}
else if (z > -0.3178794411714423215955237)
{
static const T Y = 3.49680423736572266e-01;
static const T P[] = {
-3.49729841718749014e-01,
-6.28207407760709028e+01,
-2.57226178029669171e+03,
-2.50271008623093747e+04,
1.11949239154711388e+05,
1.85684566607844318e+06,
4.80802490427638643e+06,
2.76624752134636406e+06,
};
static const T Q[] = {
1.00000000000000000e+00,
1.82717661215113000e+02,
8.00121119810280100e+03,
1.06073266717010129e+05,
3.22848993926057721e+05,
-8.05684814514171256e+05,
-2.59223192927265737e+06,
-5.61719645211570871e+05,
6.27765369292636844e+04,
};
T d = z + 0.367879441171442321595523770161460867445811;
return -d / (Y + boost::math::tools::evaluate_polynomial(P, d) / boost::math::tools::evaluate_polynomial(Q, d));
}
else if (z > -0.3578794411714423215955237701)
{
static const T Y = 5.00126481056213379e-02;
static const T  P[] = {
-5.00173570682372162e-02,
-4.44242461870072044e+01,
-9.51185533619946042e+03,
-5.88605699015429386e+05,
-1.90760843597427751e+06,
5.79797663818311404e+08,
1.11383352508459134e+10,
5.67791253678716467e+10,
6.32694500716584572e+10,
};
static const T Q[] = {
1.00000000000000000e+00,
9.08910517489981551e+02,
2.10170163753340133e+05,
1.67858612416470327e+07,
4.90435561733227953e+08,
4.54978142622939917e+09,
2.87716585708739168e+09,
-4.59414247951143131e+10,
-1.72845216404874299e+10,
};
T d = z + 0.36787944117144232159552377016146086744581113103176804;
return -d / (Y + boost::math::tools::evaluate_polynomial(P, d) / boost::math::tools::evaluate_polynomial(Q, d));
}
else
{  
const T p2 = 2 * (boost::math::constants::e<T>() * z + 1);
const T p = sqrt(p2);
return lambert_w_detail::lambert_w_singularity_series(p);
}
}

template <class T, class Policy>
inline T lambert_w0_imp(T z, const Policy& pol, const boost::integral_constant<int, 2>&)
{
static const char* function = "boost::math::lambert_w0<%1%>";
BOOST_MATH_STD_USING 

BOOST_STATIC_ASSERT_MSG(std::numeric_limits<double>::digits >= 53,
"Our double precision coefficients will be truncated, "
"please file a bug report with details of your platform's floating point types "
"- or possibly edit the coefficients to have "
"an appropriate size-suffix for 64-bit floats on your platform - L?");

if ((boost::math::isnan)(z))
{
return boost::math::policies::raise_domain_error<T>(function, "Expected a value > -e^-1 (-0.367879...) but got %1%.", z, pol);
}
if ((boost::math::isinf)(z))
{
return boost::math::policies::raise_overflow_error<T>(function, "Expected a finite value but got %1%.", z, pol);
}

if (z >= 0.05)
{
return lambert_w_positive_rational_double(z);
}
else if (z <= -0.36787944117144232159552377016146086744581113103176804) 
{
if (z < -0.36787944117144232159552377016146086744581113103176804)
{
return boost::math::policies::raise_domain_error<T>(function, "Expected z >= -e^-1 (-0.367879...) but got %1%.", z, pol);
}
return -1;
}
else
{
return lambert_w_negative_rational_double(z, pol);
}
} 


template <class T, class Policy>
inline T lambert_w0_imp(T z, const Policy& pol, const boost::integral_constant<int, 0>&)
{
static const char* function = "boost::math::lambert_w0<%1%>";
BOOST_MATH_STD_USING 

if ((boost::math::isnan)(z))
{
return boost::math::policies::raise_domain_error<T>(function, "Expected z >= -e^-1 (-0.367879...) but got %1%.", z, pol);
}
if (fabs(z) <= 0.05f)
{
return lambert_w0_small_z(z, pol);
}
if (z > (std::numeric_limits<double>::max)())
{
if ((boost::math::isinf)(z))
{
return policies::raise_overflow_error<T>(function, 0, pol);
}

T w = lambert_w0_approx(z);  
return lambert_w_halley_iterate(w, z);
}
if (z < -0.3578794411714423215955237701)
{ 
if (z <= -boost::math::constants::exp_minus_one<T>())
{
if (z == -boost::math::constants::exp_minus_one<T>())
{ 
return -1;
}
return boost::math::policies::raise_domain_error<T>(function, "Expected z >= -e^-1 (-0.367879...) but got %1%.", z, pol);
}
const T p2 = 2 * (boost::math::constants::e<T>() * z + 1);
const T p = sqrt(p2);
T w = lambert_w_detail::lambert_w_singularity_series(p);
return lambert_w_halley_iterate(w, z);
}


typedef typename policies::precision<T, Policy>::type precision_type;
typedef boost::integral_constant<bool,
(precision_type::value == 0) || (precision_type::value > 113) ?
true 
: false 
> tag_type;

T w = lambert_w0_imp(maybe_reduce_to_double(z, boost::is_constructible<double, T>()), pol, boost::integral_constant<int, 2>());

return lambert_w_maybe_halley_iterate(w, z, tag_type());

} 


template<typename T, class Policy>
T lambert_wm1_imp(const T z, const Policy&  pol)
{
BOOST_STATIC_ASSERT_MSG(!boost::is_integral<T>::value,
"Must be floating-point or fixed type (not integer type), for example: lambert_wm1(1.), not lambert_wm1(1)!");

BOOST_MATH_STD_USING 

const char* function = "boost::math::lambert_wm1<RealType>(<RealType>)"; 

if ((boost::math::isnan)(z))
{
return policies::raise_domain_error(function,
"Argument z is NaN!",
z, pol);
} 

if ((boost::math::isinf)(z))
{
return policies::raise_domain_error(function,
"Argument z is infinite!",
z, pol);
} 

if (z == static_cast<T>(0))
{ 
if (std::numeric_limits<T>::has_infinity)
{
return -std::numeric_limits<T>::infinity();
}
else
{
return -tools::max_value<T>();
}
}
if (std::numeric_limits<T>::has_denorm)
{ 
if (!(boost::math::isnormal)(z))
{ 
return policies::raise_overflow_error(function,
"Argument z =  %1% is denormalized! (must be z > (std::numeric_limits<RealType>::min)() or z == 0)",
z, pol);
}
}

if (z > static_cast<T>(0))
{ 
return policies::raise_domain_error(function,
"Argument z = %1% is out of range (z <= 0) for Lambert W-1 branch! (Try Lambert W0 branch?)",
z, pol);
}
if (z > -boost::math::tools::min_value<T>())
{ 
return policies::raise_overflow_error(function,
"Argument z = %1% is too small (z < -std::numeric_limits<T>::min so denormalized) for Lambert W-1 branch!",
z, pol);
}
if (z == -boost::math::constants::exp_minus_one<T>()) 
{ 
return -static_cast<T>(1);
}
if (z < -boost::math::constants::exp_minus_one<T>()) 
{
return policies::raise_domain_error(function,
"Argument z = %1% is out of range (z < -exp(-1) = -3.6787944... <= 0) for Lambert W-1 (or W0) branch!",
z, pol);
}
if (z < static_cast<T>(-0.35))
{ 
const T p2 = 2 * (boost::math::constants::e<T>() * z + 1);
if (p2 == 0)
{ 
return -1;
}
if (p2 > 0)
{
T w_series = lambert_w_singularity_series(T(-sqrt(p2)));
if (boost::math::tools::digits<T>() > 53)
{ 
w_series = lambert_w_detail::lambert_w_halley_iterate(w_series, z);
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_WM1_NOT_BUILTIN
std::streamsize saved_precision = std::cout.precision(std::numeric_limits<T>::max_digits10);
std::cout << "Lambert W-1 Halley updated to " << w_series << std::endl;
std::cout.precision(saved_precision);
#endif 
}
return w_series;
}
return policies::raise_domain_error(function,
"Argument z = %1% is out of range for Lambert W-1 branch. (Should not get here - please report!)",
z, pol);
} 

using lambert_w_lookup::wm1es;
using lambert_w_lookup::wm1zs;
using lambert_w_lookup::noof_wm1zs; 


if (z >= wm1zs[63]) 
{  


T guess; 
T lz = log(-z);
T llz = log(-lz);
guess = lz - llz + (llz / lz); 
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_WM1_TINY
std::streamsize saved_precision = std::cout.precision(std::numeric_limits<T>::max_digits10);
std::cout << "z = " << z << ", guess = " << guess << ", ln(-z) = " << lz << ", ln(-ln(-z) = " << llz << ", llz/lz = " << (llz / lz) << std::endl;
int d10 = policies::digits_base10<T, Policy>(); 
int d2 = policies::digits<T, Policy>(); 
std::cout << "digits10 = " << d10 << ", digits2 = " << d2 
<< std::endl;
std::cout.precision(saved_precision);
#endif 
if (policies::digits<T, Policy>() < 12)
{ 
return guess;
}
T result = lambert_w_detail::lambert_w_halley_iterate(guess, z);
return result;

} 

if (boost::math::tools::digits<T>() > 53)
{ 
using boost::math::policies::precision;
using boost::math::policies::digits10;
using boost::math::policies::digits2;
using boost::math::policies::policy;
T double_approx(static_cast<T>(lambert_wm1_imp(must_reduce_to_double(z, boost::is_constructible<double, T>()), policy<digits2<50> >())));
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_WM1_NOT_BUILTIN
std::streamsize saved_precision = std::cout.precision(std::numeric_limits<T>::max_digits10);
std::cout << "Lambert_wm1 Argument Type " << typeid(T).name() << " approximation double = " << double_approx << std::endl;
std::cout.precision(saved_precision);
#endif 
T result = lambert_w_halley_iterate(double_approx, z);
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_WM1
std::streamsize saved_precision = std::cout.precision(std::numeric_limits<T>::max_digits10);
std::cout << "Result " << typeid(T).name() << " precision Halley refinement =    " << result << std::endl;
std::cout.precision(saved_precision);
#endif 
return result;
} 
else 
{ 
using namespace boost::math::lambert_w_detail::lambert_w_lookup;
int n = 2;
if (wm1zs[n - 1] > z)
{
goto bisect;
}
for (int j = 1; j <= 5; ++j)
{
n *= 2;
if (wm1zs[n - 1] > z)
{
goto overshot;
}
}
return policies::raise_domain_error(function,
"Argument z = %1% is too small (< -1.026439e-26) (logic error - please report!)",
z, pol);
overshot:
{
int nh = n / 2;
for (int j = 1; j <= 5; ++j)
{
nh /= 2; 
if (nh <= 0)
{
break; 
}
if (wm1zs[n - nh - 1] > z)
{
n -= nh;
}
}
}
bisect:
--n;  
#ifdef BOOST_MATH_INSTRUMENT_LAMBERT_WM1_LOOKUP
std::streamsize saved_precision = std::cout.precision(std::numeric_limits<T>::max_digits10);
std::cout << "Result lookup W-1(" << z << ") bisection between wm1zs[" << n - 1 << "] = " << wm1zs[n - 1] << " and wm1zs[" << n << "] = " << wm1zs[n]
<< ", bisect mean = " << (wm1zs[n - 1] + wm1zs[n]) / 2 << std::endl;
std::cout.precision(saved_precision);
#endif 

int bisections = 11; 
if (n >= 8)
{
bisections = 8;
}
else if (n >= 3)
{
bisections = 9;
}
else if (n >= 2)
{
bisections = 10;
}
using lambert_w_lookup::halves;
using lambert_w_lookup::sqrtwm1s;

typedef typename mpl::if_c<boost::is_constructible<lookup_t, T>::value, lookup_t, T>::type calc_type;

calc_type w = -static_cast<calc_type>(n); 
calc_type y = static_cast<calc_type>(z * wm1es[n - 1]); 
for (int j = 0; j < bisections; ++j)
{ 
calc_type wj = w - halves[j]; 
calc_type yj = y * sqrtwm1s[j]; 
if (wj < yj)
{
w = wj;
y = yj;
}
} 
return static_cast<T>(schroeder_update(w, y)); 

}
} 
} 


template <class T, class Policy>
inline
typename boost::math::tools::promote_args<T>::type
lambert_w0(T z, const Policy& pol)
{
typedef typename tools::promote_args<T>::type result_type;

typedef typename policies::precision<result_type, Policy>::type precision_type;
typedef boost::integral_constant<int,
(precision_type::value == 0) || (precision_type::value > 53) ?
0  
: (precision_type::value <= 24) ? 1 
: 2  
> tag_type;

return lambert_w_detail::lambert_w0_imp(result_type(z), pol, tag_type()); 
} 

template <class T>
inline
typename tools::promote_args<T>::type
lambert_w0(T z)
{
typedef typename tools::promote_args<T>::type result_type;

typedef typename policies::precision<result_type, policies::policy<> >::type precision_type;
typedef boost::integral_constant<int,
(precision_type::value == 0) || (precision_type::value > 53) ?
0  
: (precision_type::value <= 24) ? 1 
: 2  
> tag_type;
return lambert_w_detail::lambert_w0_imp(result_type(z),  policies::policy<>(), tag_type());
} 


template <class T, class Policy>
inline
typename tools::promote_args<T>::type
lambert_wm1(T z, const Policy& pol)
{
typedef typename tools::promote_args<T>::type result_type;
return lambert_w_detail::lambert_wm1_imp(result_type(z), pol); 
}

template <class T>
inline
typename tools::promote_args<T>::type
lambert_wm1(T z)
{
typedef typename tools::promote_args<T>::type result_type;
return lambert_w_detail::lambert_wm1_imp(result_type(z), policies::policy<>());
} 

template <class T, class Policy>
inline typename tools::promote_args<T>::type
lambert_w0_prime(T z, const Policy& pol)
{
typedef typename tools::promote_args<T>::type result_type;
using std::numeric_limits;
if (z == 0)
{
return static_cast<result_type>(1);
}
if (z == - boost::math::constants::exp_minus_one<result_type>())
{
return numeric_limits<result_type>::has_infinity ? numeric_limits<result_type>::infinity() : boost::math::tools::max_value<result_type>();
}
result_type w = lambert_w0(result_type(z), pol);
return w / (z * (1 + w));
} 

template <class T>
inline typename tools::promote_args<T>::type
lambert_w0_prime(T z)
{
return lambert_w0_prime(z, policies::policy<>());
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type
lambert_wm1_prime(T z, const Policy& pol)
{
using std::numeric_limits;
typedef typename tools::promote_args<T>::type result_type;
if (z == 0 || z == - boost::math::constants::exp_minus_one<result_type>())
{
return numeric_limits<result_type>::has_infinity ? -numeric_limits<result_type>::infinity() : -boost::math::tools::max_value<result_type>();
}

result_type w = lambert_wm1(z, pol);
return w/(z*(1+w));
} 

template <class T>
inline typename tools::promote_args<T>::type
lambert_wm1_prime(T z)
{
return lambert_wm1_prime(z, policies::policy<>());
}

}} 

#endif 

