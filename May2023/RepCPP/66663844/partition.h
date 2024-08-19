#ifndef PARTITION_H
#define PARTITION_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "block_swap.h"

namespace detail {
template <typename Iterator, typename Pred>
Iterator partition(Iterator begin, Iterator end, Pred pred) {
auto d = end - begin;

constexpr size_t simple_partition_threshold = 8192;
if (d <= simple_partition_threshold) {
return std::stable_partition(begin, end, pred);
}

Iterator it1, it2, middle = begin + d / 2;

constexpr size_t parallel_threshold = 16384;
#pragma omp taskgroup
{
#pragma omp task final(middle - begin < parallel_threshold) default(none) firstprivate(begin, middle, pred) shared(it1)
{ it1 = detail::partition(begin, middle, pred); }
#pragma omp task final(end - middle < parallel_threshold) default(none) firstprivate(middle, end, pred) shared(it2)
{ it2 = detail::partition(middle, end, pred); }
}

block_swap(it1, middle, it2);
return it1 + (it2 - middle);
}
}

template <typename Iterator, typename Pred>
Iterator partition(Iterator begin, Iterator end, Pred pred) {
Iterator it;
#pragma omp parallel if(omp_get_level() == 0)
{
#pragma omp single nowait
{ it = detail::partition(begin, end, pred); }
}
return it;
}

#endif 
