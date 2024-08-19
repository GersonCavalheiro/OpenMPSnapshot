#pragma once

#include <type_traits> 
#include <complex> 

#include "status.hxx" 
#ifndef NO_UNIT_TESTS
#include <cstdio> 
#endif  

template <typename T> struct is_complex_t                  : public std::false_type {};
template <typename T> struct is_complex_t<std::complex<T>> : public std::true_type  {};
template <typename T> constexpr bool is_complex(T const x=0) { return is_complex_t<T>::value; } 

template <typename complex_t> char const * complex_name(complex_t const x=0);
template <> inline char const * complex_name<double>(double const x) { return "double"; }
template <> inline char const * complex_name<float> (float  const x) { return "float"; }
template <> inline char const * complex_name<std::complex<double>>(std::complex<double> const x) { return "complex<double>"; }
template <> inline char const * complex_name<std::complex<float>> (std::complex<float>  const x) { return "complex<float>"; }

template <typename real_t> real_t conjugate(real_t const x);
template <> inline double conjugate<double>(double const x) { return x; }
template <> inline float  conjugate<float> (float  const x) { return x; }
template <typename real_t>
std::complex<real_t> inline conjugate(std::complex<real_t> const x) { return std::conj(x); }

template <typename complex_t, typename real_t> inline
complex_t to_complex_t(std::complex<real_t> const x); 
template <> inline std::complex<double> to_complex_t(std::complex<double> const x) { return x; }
template <> inline std::complex<float>  to_complex_t(std::complex<float>  const x) { return x; }
template <> inline double               to_complex_t(std::complex<double> const x) { return x.real(); }
template <> inline float                to_complex_t(std::complex<float>  const x) { return x.real(); }
template <> inline float                to_complex_t(std::complex<double> const x) { return x.real(); }
template <> inline double               to_complex_t(std::complex<float>  const x) { return x.real(); }
template <> inline std::complex<float>  to_complex_t(std::complex<double> const x) { return std::complex<float> (x.real(), x.imag()); }
template <> inline std::complex<double> to_complex_t(std::complex<float>  const x) { return std::complex<double>(x.real(), x.imag()); }

inline std::complex<double> to_double_complex_t(std::complex<double> const x) { return x; }
inline std::complex<double> to_double_complex_t(std::complex<float>  const x) { return std::complex<double>(x.real(), x.imag()); }
inline double               to_double_complex_t(double               const x) { return x; }
inline double               to_double_complex_t(float                const x) { return double(x); }

namespace complex_tools {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename complex_t>
inline status_t test_complex(bool const expect_complex, int const echo=0) {
bool const is = is_complex<complex_t>();
if (echo > 0) std::printf("# %s is_complex<%s>() = %s\n", __func__,
complex_name<complex_t>(), is?"true":"false");
if (echo > 1) std::printf("# %s typeof(conjugate(%s x)) = %s\n", __func__,
complex_name<complex_t>(),
complex_name<decltype(conjugate(complex_t(1)))>());
return (expect_complex != is);
} 

template <typename real_t>
inline status_t test_conjugate(int const echo=0) {
std::complex<real_t> x(0, 1); 
auto const xpxc = x + conjugate(x);
return (0 != xpxc.real()) + (0 != xpxc.imag());
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_complex<std::complex<double>>(true, echo);
stat += test_complex<std::complex<float>> (true, echo);
stat += test_complex<double>(false, echo);
stat += test_complex<float> (false, echo);
stat += test_conjugate<double>(echo);
stat += test_conjugate<float> (echo);
return stat;
} 

#endif 

} 
