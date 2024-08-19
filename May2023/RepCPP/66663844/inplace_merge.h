#ifndef INPLACE_MERGE_H
#define INPLACE_MERGE_H

#include <algorithm>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "block_swap.h"

namespace detail {

template <typename Iterator, typename Cmp>
void inplace_merge(Iterator a, Iterator b, Iterator c, Cmp cmp) {
auto d1 = b - a;
auto d2 = c - b;
if (d1 + d2 <= 1) return;

typedef typename std::iterator_traits<Iterator>::value_type T;
constexpr size_t simple_merge_threshold = 4096 / sizeof(T);
if (d1 + d2 <= simple_merge_threshold) {
T tmp[simple_merge_threshold];
std::merge(a, b, b, c, tmp);
std::copy(tmp, tmp + (d1 + d2), a);
return;
}

Iterator m, p1, p2;
if (d1 > d2) {
auto k = a + d1 / 2;
auto it = std::lower_bound(b, c, *k, cmp);

m = k + (it - b);
p1 = k;
p2 = it;
} else {
auto k = b + d2 / 2;
auto it = std::upper_bound(a, b, *k, cmp);

m = it + (k - b);
p1 = it;
p2 = k + 1;
}

block_swap(p1, b, p2);

constexpr size_t parallel_threshold = 16384;
#pragma omp task final(m - a < parallel_threshold) default(none) firstprivate(a, p1, m, cmp)
{ detail::inplace_merge(a, p1, m, cmp); }
#pragma omp task final(c - m < parallel_threshold) default(none) firstprivate(m, p2, c, cmp)
{ detail::inplace_merge(std::next(m), p2, c, cmp); }
}

} 

template <typename Iterator, typename Cmp = std::less<typename std::iterator_traits<Iterator>::value_type> >
void inplace_merge(Iterator a, Iterator b, Iterator c, Cmp cmp = Cmp()) {
#pragma omp parallel if(omp_get_level() == 0)
{
#pragma omp single nowait
{ detail::inplace_merge(a, b, c, cmp); }
}
}

#endif 
