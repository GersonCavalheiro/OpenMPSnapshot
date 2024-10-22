#pragma once
#include <type_traits>
#include <cmath>
#include <random>

using std::is_floating_point;
using std::uniform_int_distribution;
using std::ceil;
using std::sqrt;





template <class T>
inline T ceilDiv(T x, T y) {
if (is_floating_point<T>::value) return ceil(x/y);
return (x + y-1) / y;
}





template <typename T>
inline int sgn(T x) {
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





template <class T>
inline bool isPrime(T x) {
if (x<=3) return x>1;
if (x % 2==0 || x % 3==0) return false;
for (T i=6, I=T(sqrt(x))+1; i<=I; i+=6)
if (x % (i-1)==0 || x % (i+1)==0) return false;
return true;
}

template <class T>
inline T nextPrime(T x) {
while (true)
if (isPrime(++x)) return x;
}

template <class T, class R>
inline T randomPrime(T begin, T end, R& rnd) {
uniform_int_distribution<T> dis(begin, end);
for (int i=128; i>0; --i) {
T a = dis(rnd);
if (isPrime(a)) return a;
}
return end-1;
}
