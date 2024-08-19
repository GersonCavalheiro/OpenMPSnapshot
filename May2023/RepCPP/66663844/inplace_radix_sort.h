#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "insertion_sort.h"
#include "partition.h"

namespace detail {
template <int N> struct MaskType {};
template <> struct MaskType<1> { typedef uint8_t  Type; };
template <> struct MaskType<2> { typedef uint16_t Type; };
template <> struct MaskType<4> { typedef uint32_t Type; };
template <> struct MaskType<8> { typedef uint64_t Type; };

template <typename Iterator, typename Mask>
void inplace_radix_sort(Iterator begin, Iterator end, int bits) {
constexpr size_t simple_sort_threshold = 32;
if (end - begin <= simple_sort_threshold) {
insertion_sort(begin, end);
return;
}

if (bits <= 0) return;

typedef typename std::iterator_traits<Iterator>::value_type T;
Mask mask = Mask(1) << (bits - 1);
auto it = detail::partition(begin, end, [=] (const T& t) { return (t & mask) == 0; });

constexpr size_t parallel_threshold = 4096;
#pragma omp task final(end - it < parallel_threshold) default(none) firstprivate(it, end, bits)
{ detail::inplace_radix_sort<Iterator, Mask>(it, end, bits - 1); }
#pragma omp task final(it - begin < parallel_threshold) default(none) firstprivate(begin, it, bits)
{ detail::inplace_radix_sort<Iterator, Mask>(begin, it, bits - 1); }
}
}

template <typename Iterator, typename Mask = typename detail::MaskType<sizeof(typename std::iterator_traits<Iterator>::value_type)>::Type>
void inplace_radix_sort(Iterator begin, Iterator end, int bits = sizeof(typename std::iterator_traits<Iterator>::value_type) * 8) {
#pragma omp parallel if(omp_get_level() == 0)
{
#pragma omp single nowait
{ detail::inplace_radix_sort<Iterator, Mask>(begin, end, bits); }
}
}

#endif 
