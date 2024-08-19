#pragma once
#include <vector>
#include "_main.hxx"
#include "multiply.hxx"

using std::vector;




template <class T>
float multiplyOpenmp(T *a, const T *x, const T *y, int N, const MultiplyOptions& o={}) {
return measureDuration([&] { multiplyOmp(a, x, y, N); }, o.repeat);
}

template <class T>
float multiplyOpenmp(vector<T>& a, const vector<T>& x, const vector<T>& y, const MultiplyOptions& o={}) {
return multiplyOpenmp(a.data(), x.data(), y.data(), x.size(), o);
}
