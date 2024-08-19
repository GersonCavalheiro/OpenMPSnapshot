#pragma once
#include <vector>
#include "_main.hxx"
#include "multiply.hxx"

using std::vector;




template <class T>
float multiplySeq(T *a, const T *x, const T *y, int N, const MultiplyOptions& o={}) {
return measureDuration([&] { multiply(a, x, y, N); }, o.repeat);
}

template <class T>
float multiplySeq(vector<T>& a, const vector<T>& x, const vector<T>& y, const MultiplyOptions& o={}) {
return multiplySeq(a.data(), x.data(), y.data(), x.size(), o);
}
