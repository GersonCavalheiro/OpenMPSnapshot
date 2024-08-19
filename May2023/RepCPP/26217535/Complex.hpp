

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/math/FloatEqualExact.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <type_traits>

namespace alpaka
{
template<typename T>
class Complex
{
public:
static_assert(std::is_floating_point_v<T>);

using value_type = T;

constexpr ALPAKA_FN_HOST_ACC Complex(T const& real = T{}, T const& imag = T{}) : m_real(real), m_imag(imag)
{
}

constexpr Complex(Complex const& other) = default;

template<typename U>
constexpr ALPAKA_FN_HOST_ACC Complex(Complex<U> const& other)
: m_real(static_cast<T>(other.real()))
, m_imag(static_cast<T>(other.imag()))
{
}

constexpr ALPAKA_FN_HOST_ACC Complex(std::complex<T> const& other) : m_real(other.real()), m_imag(other.imag())
{
}

constexpr ALPAKA_FN_HOST_ACC operator std::complex<T>() const
{
return std::complex<T>{m_real, m_imag};
}

Complex& operator=(Complex const&) = default;

constexpr ALPAKA_FN_HOST_ACC T real() const
{
return m_real;
}

constexpr ALPAKA_FN_HOST_ACC void real(T value)
{
m_real = value;
}

constexpr ALPAKA_FN_HOST_ACC T imag() const
{
return m_imag;
}

constexpr ALPAKA_FN_HOST_ACC void imag(T value)
{
m_imag = value;
}

ALPAKA_FN_HOST_ACC Complex& operator+=(T const& other)
{
m_real += other;
return *this;
}

template<typename U>
ALPAKA_FN_HOST_ACC Complex& operator+=(Complex<U> const& other)
{
m_real += static_cast<T>(other.real());
m_imag += static_cast<T>(other.imag());
return *this;
}

ALPAKA_FN_HOST_ACC Complex& operator-=(T const& other)
{
m_real -= other;
return *this;
}

template<typename U>
ALPAKA_FN_HOST_ACC Complex& operator-=(Complex<U> const& other)
{
m_real -= static_cast<T>(other.real());
m_imag -= static_cast<T>(other.imag());
return *this;
}

ALPAKA_FN_HOST_ACC Complex& operator*=(T const& other)
{
m_real *= other;
m_imag *= other;
return *this;
}

template<typename U>
ALPAKA_FN_HOST_ACC Complex& operator*=(Complex<U> const& other)
{
auto const newReal = m_real * static_cast<T>(other.real()) - m_imag * static_cast<T>(other.imag());
auto const newImag = m_imag * static_cast<T>(other.real()) + m_real * static_cast<T>(other.imag());
m_real = newReal;
m_imag = newImag;
return *this;
}

ALPAKA_FN_HOST_ACC Complex& operator/=(T const& other)
{
m_real /= other;
m_imag /= other;
return *this;
}

template<typename U>
ALPAKA_FN_HOST_ACC Complex& operator/=(Complex<U> const& other)
{
return *this *= Complex{
static_cast<T>(other.real() / (other.real() * other.real() + other.imag() * other.imag())),
static_cast<T>(-other.imag() / (other.real() * other.real() + other.imag() * other.imag()))};
}

private:
T m_real, m_imag;
};


template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator+(Complex<T> const& val)
{
return val;
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator-(Complex<T> const& val)
{
return Complex<T>{-val.real(), -val.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator+(Complex<T> const& lhs, Complex<T> const& rhs)
{
return Complex<T>{lhs.real() + rhs.real(), lhs.imag() + rhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator+(Complex<T> const& lhs, T const& rhs)
{
return Complex<T>{lhs.real() + rhs, lhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator+(T const& lhs, Complex<T> const& rhs)
{
return Complex<T>{lhs + rhs.real(), rhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator-(Complex<T> const& lhs, Complex<T> const& rhs)
{
return Complex<T>{lhs.real() - rhs.real(), lhs.imag() - rhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator-(Complex<T> const& lhs, T const& rhs)
{
return Complex<T>{lhs.real() - rhs, lhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator-(T const& lhs, Complex<T> const& rhs)
{
return Complex<T>{lhs - rhs.real(), -rhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator*(Complex<T> const& lhs, Complex<T> const& rhs)
{
return Complex<T>{
lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
lhs.imag() * rhs.real() + lhs.real() * rhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator*(Complex<T> const& lhs, T const& rhs)
{
return Complex<T>{lhs.real() * rhs, lhs.imag() * rhs};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator*(T const& lhs, Complex<T> const& rhs)
{
return Complex<T>{lhs * rhs.real(), lhs * rhs.imag()};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator/(Complex<T> const& lhs, Complex<T> const& rhs)
{
return Complex<T>{
(lhs.real() * rhs.real() + lhs.imag() * rhs.imag()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag()),
(lhs.imag() * rhs.real() - lhs.real() * rhs.imag()) / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag())};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator/(Complex<T> const& lhs, T const& rhs)
{
return Complex<T>{lhs.real() / rhs, lhs.imag() / rhs};
}

template<typename T>
ALPAKA_FN_HOST_ACC Complex<T> operator/(T const& lhs, Complex<T> const& rhs)
{
return Complex<T>{
lhs * rhs.real() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag()),
-lhs * rhs.imag() / (rhs.real() * rhs.real() + rhs.imag() * rhs.imag())};
}

template<typename T>
constexpr ALPAKA_FN_HOST_ACC bool operator==(Complex<T> const& lhs, Complex<T> const& rhs)
{
return math::floatEqualExactNoWarning(lhs.real(), rhs.real())
&& math::floatEqualExactNoWarning(lhs.imag(), rhs.imag());
}

template<typename T>
constexpr ALPAKA_FN_HOST_ACC bool operator==(Complex<T> const& lhs, T const& rhs)
{
return math::floatEqualExactNoWarning(lhs.real(), rhs)
&& math::floatEqualExactNoWarning(lhs.imag(), static_cast<T>(0));
}

template<typename T>
constexpr ALPAKA_FN_HOST_ACC bool operator==(T const& lhs, Complex<T> const& rhs)
{
return math::floatEqualExactNoWarning(lhs, rhs.real())
&& math::floatEqualExactNoWarning(static_cast<T>(0), rhs.imag());
}

template<typename T>
constexpr ALPAKA_FN_HOST_ACC bool operator!=(Complex<T> const& lhs, Complex<T> const& rhs)
{
return !(lhs == rhs);
}

template<typename T>
constexpr ALPAKA_FN_HOST_ACC bool operator!=(Complex<T> const& lhs, T const& rhs)
{
return !math::floatEqualExactNoWarning(lhs.real(), rhs)
|| !math::floatEqualExactNoWarning(lhs.imag(), static_cast<T>(0));
}

template<typename T>
constexpr ALPAKA_FN_HOST_ACC bool operator!=(T const& lhs, Complex<T> const& rhs)
{
return !math::floatEqualExactNoWarning(lhs, rhs.real())
|| !math::floatEqualExactNoWarning(static_cast<T>(0), rhs.imag());
}


template<typename T, typename TChar, typename TTraits>
std::basic_ostream<TChar, TTraits>& operator<<(std::basic_ostream<TChar, TTraits>& os, Complex<T> const& x)
{
os << x.operator std::complex<T>();
return os;
}

template<typename T, typename TChar, typename TTraits>
std::basic_istream<TChar, TTraits>& operator>>(std::basic_istream<TChar, TTraits>& is, Complex<T> const& x)
{
std::complex<T> z;
is >> z;
x = z;
return is;
}


ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC T abs(Complex<T> const& x)
{
return std::abs(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> acos(Complex<T> const& x)
{
return std::acos(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> acosh(Complex<T> const& x)
{
return std::acosh(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC T arg(Complex<T> const& x)
{
return std::arg(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> asin(Complex<T> const& x)
{
return std::asin(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> asinh(Complex<T> const& x)
{
return std::asinh(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> atan(Complex<T> const& x)
{
return std::atan(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> atanh(Complex<T> const& x)
{
return std::atanh(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> conj(Complex<T> const& x)
{
return std::conj(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> cos(Complex<T> const& x)
{
return std::cos(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> cosh(Complex<T> const& x)
{
return std::cosh(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> exp(Complex<T> const& x)
{
return std::exp(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> log(Complex<T> const& x)
{
return std::log(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> log10(Complex<T> const& x)
{
return std::log10(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC T norm(Complex<T> const& x)
{
return std::norm(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> polar(T const& r, T const& theta = T())
{
return std::polar(r, theta);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T, typename U>
constexpr ALPAKA_FN_HOST_ACC auto pow(Complex<T> const& x, Complex<U> const& y)
{
auto const result = std::pow(std::complex<T>(x), std::complex<U>(y));
using ValueType = typename decltype(result)::value_type;
return Complex<ValueType>(result);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T, typename U>
constexpr ALPAKA_FN_HOST_ACC auto pow(Complex<T> const& x, U const& y)
{
return pow(x, Complex<U>(y));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T, typename U>
constexpr ALPAKA_FN_HOST_ACC auto pow(T const& x, Complex<U> const& y)
{
return pow(Complex<T>(x), y);
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> proj(Complex<T> const& x)
{
return std::proj(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> sin(Complex<T> const& x)
{
return std::sin(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> sinh(Complex<T> const& x)
{
return std::sinh(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> sqrt(Complex<T> const& x)
{
return std::sqrt(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> tan(Complex<T> const& x)
{
return std::tan(std::complex<T>(x));
}

ALPAKA_NO_HOST_ACC_WARNING
template<typename T>
constexpr ALPAKA_FN_HOST_ACC Complex<T> tanh(Complex<T> const& x)
{
return std::tanh(std::complex<T>(x));
}


} 
