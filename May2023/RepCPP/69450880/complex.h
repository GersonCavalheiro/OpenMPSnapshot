



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <cmath>
#include <complex>
#include <sstream>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#  define HYDRA_THRUST_STD_COMPLEX_REAL(z) \
reinterpret_cast< \
const typename hydra_thrust::detail::remove_reference<decltype(z)>::type::value_type (&)[2] \
>(z)[0]
#  define HYDRA_THRUST_STD_COMPLEX_IMAG(z) \
reinterpret_cast< \
const typename hydra_thrust::detail::remove_reference<decltype(z)>::type::value_type (&)[2] \
>(z)[1]
#  define HYDRA_THRUST_STD_COMPLEX_DEVICE __device__
#else
#  define HYDRA_THRUST_STD_COMPLEX_REAL(z) (z).real()
#  define HYDRA_THRUST_STD_COMPLEX_IMAG(z) (z).imag()
#  define HYDRA_THRUST_STD_COMPLEX_DEVICE
#endif

namespace hydra_thrust
{








namespace detail
{

template <typename T, std::size_t Align>
struct complex_storage;

#if __cplusplus >= 201103L                                                    \
&& (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC)                       \
&& (HYDRA_THRUST_GCC_VERSION >= 40800)
template <typename T, std::size_t Align>
struct complex_storage
{
struct alignas(Align) type { T x; T y; };
};
#elif  (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC)                    \
|| (   (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_GCC)                 \
&& (HYDRA_THRUST_GCC_VERSION < 40600))

#if (HYDRA_THRUST_HOST_COMPILER == HYDRA_THRUST_HOST_COMPILER_MSVC)
#define HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(X)                   \
template <typename T>                                                   \
struct complex_storage<T, X>                                            \
{                                                                       \
__declspec(align(X)) struct type { T x; T y; };                       \
};                                                                      \

#else
#define HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(X)                   \
template <typename T>                                                   \
struct complex_storage<T, X>                                            \
{                                                                       \
struct type { T x; T y; } __attribute__((aligned(X)));                \
};                                                                      \

#endif

template <typename T, std::size_t Align>
struct complex_storage
{
T x; T y;
};

HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(1);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(2);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(4);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(8);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(16);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(32);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(64);
HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION(128);

#undef HYDRA_THRUST_DEFINE_COMPLEX_STORAGE_SPECIALIZATION
#else
template <typename T, std::size_t Align>
struct complex_storage
{
struct type { T x; T y; } __attribute__((aligned(Align)));
};
#endif

} 


template <typename T>
struct complex
{
public:


typedef T value_type;






__host__ __device__
complex(const T& re);


__host__ __device__
complex(const T& re, const T& im);

#if HYDRA_THRUST_CPP_DIALECT >= 2011

complex() = default;


complex(const complex<T>& z) = default;
#else

__host__ __device__
complex();


__host__ __device__
complex(const complex<T>& z);
#endif


template <typename U>
__host__ __device__
complex(const complex<U>& z);


__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
complex(const std::complex<T>& z);


template <typename U>
__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
complex(const std::complex<U>& z);






__host__ __device__
complex& operator=(const T& re);

#if HYDRA_THRUST_CPP_DIALECT >= 2011

complex& operator=(const complex<T>& z) = default;
#else

__host__ __device__
complex& operator=(const complex<T>& z);
#endif


template <typename U>
__host__ __device__
complex& operator=(const complex<U>& z);


__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
complex& operator=(const std::complex<T>& z);


template <typename U>
__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
complex& operator=(const std::complex<U>& z);





template <typename U>
__host__ __device__
complex<T>& operator+=(const complex<U>& z);


template <typename U>
__host__ __device__
complex<T>& operator-=(const complex<U>& z);


template <typename U>
__host__ __device__
complex<T>& operator*=(const complex<U>& z);


template <typename U>
__host__ __device__
complex<T>& operator/=(const complex<U>& z);


template <typename U>
__host__ __device__
complex<T>& operator+=(const U& z);


template <typename U>
__host__ __device__
complex<T>& operator-=(const U& z);


template <typename U>
__host__ __device__
complex<T>& operator*=(const U& z);


template <typename U>
__host__ __device__
complex<T>& operator/=(const U& z);






__host__ __device__
T real() const volatile { return data.x; }


__host__ __device__
T imag() const volatile { return data.y; }


__host__ __device__
T real() const { return data.x; }


__host__ __device__
T imag() const { return data.y; }






__host__ __device__
void real(T re) volatile { data.x = re; }


__host__ __device__
void imag(T im) volatile { data.y = im; }


__host__ __device__
void real(T re) { data.x = re; }


__host__ __device__
void imag(T im) { data.y = im; }






__host__
operator std::complex<T>() const { return std::complex<T>(real(), imag()); }

private:
typename detail::complex_storage<T, sizeof(T) * 2>::type data;
};





template<typename T>
__host__ __device__
T abs(const complex<T>& z);


template <typename T>
__host__ __device__
T arg(const complex<T>& z);


template <typename T>
__host__ __device__
T norm(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> conj(const complex<T>& z);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
polar(const T0& m, const T1& theta = T1());


template <typename T>
__host__ __device__
complex<T> proj(const T& z);






template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const complex<T0>& x, const T1& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator+(const T0& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const complex<T0>& x, const T1& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator-(const T0& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const complex<T0>& x, const T1& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator*(const T0& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const complex<T0>& x, const T1& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
operator/(const T0& x, const complex<T1>& y);






template <typename T>
__host__ __device__
complex<T>
operator+(const complex<T>& y);


template <typename T>
__host__ __device__
complex<T>
operator-(const complex<T>& y);






template <typename T>
__host__ __device__
complex<T> exp(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> log(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> log10(const complex<T>& z);






template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const T1& y);


template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const T0& x, const complex<T1>& y);


template <typename T>
__host__ __device__
complex<T> sqrt(const complex<T>& z);





template <typename T>
__host__ __device__
complex<T> cos(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> sin(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> tan(const complex<T>& z);






template <typename T>
__host__ __device__
complex<T> cosh(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> sinh(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> tanh(const complex<T>& z);






template <typename T>
__host__ __device__
complex<T> acos(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> asin(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> atan(const complex<T>& z);






template <typename T>
__host__ __device__
complex<T> acosh(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> asinh(const complex<T>& z);


template <typename T>
__host__ __device__
complex<T> atanh(const complex<T>& z);






template <typename T, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const complex<T>& z);


template <typename T, typename CharT, typename Traits>
__host__
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, complex<T>& z);






template <typename T0, typename T1>
__host__ __device__
bool operator==(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
bool operator==(const complex<T0>& x, const std::complex<T1>& y);


template <typename T0, typename T1>
__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
bool operator==(const std::complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
bool operator==(const T0& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
bool operator==(const complex<T0>& x, const T1& y);


template <typename T0, typename T1>
__host__ __device__
bool operator!=(const complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
bool operator!=(const complex<T0>& x, const std::complex<T1>& y);


template <typename T0, typename T1>
__host__ HYDRA_THRUST_STD_COMPLEX_DEVICE
bool operator!=(const std::complex<T0>& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
bool operator!=(const T0& x, const complex<T1>& y);


template <typename T0, typename T1>
__host__ __device__
bool operator!=(const complex<T0>& x, const T1& y);

} 

#include <hydra/detail/external/hydra_thrust/detail/complex/complex.inl>

#undef HYDRA_THRUST_STD_COMPLEX_REAL
#undef HYDRA_THRUST_STD_COMPLEX_IMAG
#undef HYDRA_THRUST_STD_COMPLEX_DEVICE





