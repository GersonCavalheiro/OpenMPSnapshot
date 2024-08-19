#pragma once
#include <vector>
#include "_main.hxx"
#include "sum.hxx"

using std::vector;




template <class T>
SumResult<T> sumOpenmp(const T *x, int N, const SumOptions& o={}) {
T a = T(); float t = measureDuration([&] { a = sumOmp(x, N); }, o.repeat);
return {a, t};
}

template <class T>
SumResult<T> sumOpenmp(const vector<T>& x, const SumOptions& o={}) {
return sumOpenmp(x.data(), x.size(), o);
}
