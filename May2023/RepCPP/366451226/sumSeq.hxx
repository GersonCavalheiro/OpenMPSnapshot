#pragma once
#include <vector>
#include "sum.hxx"

using std::vector;




template <class T>
SumResult<T> sumSeq(const T *x, int N, const SumOptions& o={}) {
T a = T(); float t = measureDuration([&]() { a = sum(x, N); }, o.repeat);
return {a, t};
}

template <class T>
SumResult<T> sumSeq(const vector<T>& x, const SumOptions& o={}) {
return sumSeq(x.data(), x.size(), o);
}
