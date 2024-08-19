#ifndef INPLACE_MERGE_SORT_H
#define INPLACE_MERGE_SORT_H

#include <algorithm>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "inplace_merge.h"
#include "shell_sort.h"

namespace detail {

template <typename Iterator, typename Cmp>
void inplace_merge_sort(Iterator begin, Iterator end, Cmp cmp) {
auto d = end - begin;

constexpr size_t simple_sort_threshold = 4096;
if (d < simple_sort_threshold) {
std::stable_sort(begin, end, cmp);
return;
}

auto middle = begin + d / 2;

constexpr size_t parallel_threshold = 8192;
#pragma omp taskgroup
{
#pragma omp task final(middle - begin < parallel_threshold) default(none) firstprivate(begin, middle, cmp)
{ detail::inplace_merge_sort(begin, middle, cmp); }
#pragma omp task final(end - middle < parallel_threshold) default(none) firstprivate(middle, end, cmp)
{ detail::inplace_merge_sort(middle, end, cmp); }
}

detail::inplace_merge(begin, middle, end, cmp);
}

} 

template <typename Iterator, typename Cmp = std::less<typename std::iterator_traits<Iterator>::value_type> >
void inplace_merge_sort(Iterator begin, Iterator end, Cmp cmp = Cmp()) {
#pragma omp parallel if(omp_get_level() == 0)
{
#pragma omp single nowait
{ detail::inplace_merge_sort(begin, end, cmp); }
}
}

#endif 
