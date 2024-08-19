#pragma once
#include <cmath>

using std::ceil;





template <class T>
T coalesce(T x, T d=T()) {
return x!=T()? x : d;
}





template <class T>
T ceilDiv(T x, T y) { return (x + y-1) / y; }
template <>
float ceilDiv<float>(float x, float y) { return ceil(x/y); }
template <>
double ceilDiv<double>(double x, double y) { return ceil(x/y); }





template <typename T>
int sgn(T x) {
return (T() < x) - (x < T());
}





template <class T>
constexpr bool isPow2(T x) noexcept {
return !(x & (x-1));
}


template <class T>
constexpr T prevPow2(T x) noexcept {
return 1 << T(log2(x));
}


template <class T>
constexpr T nextPow2(T x) noexcept {
return 1 << T(ceil(log2(x)));
}
