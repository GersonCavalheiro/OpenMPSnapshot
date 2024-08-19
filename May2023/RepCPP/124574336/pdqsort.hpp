
#ifndef BOOST_MOVE_ALGO_PDQSORT_HPP
#define BOOST_MOVE_ALGO_PDQSORT_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/detail/config_begin.hpp>
#include <boost/move/detail/workaround.hpp>
#include <boost/move/utility_core.hpp>
#include <boost/move/algo/detail/insertion_sort.hpp>
#include <boost/move/algo/detail/heap_sort.hpp>
#include <boost/move/detail/iterator_traits.hpp>

#include <boost/move/adl_move_swap.hpp>
#include <cstddef>

namespace boost {
namespace movelib {

namespace pdqsort_detail {

template<class T1, class T2>
struct pair
{
pair()
{}

pair(const T1 &t1, const T2 &t2)
: first(t1), second(t2)
{}

T1 first;
T2 second;
};

enum {
insertion_sort_threshold = 24,

ninther_threshold = 128,

partial_insertion_sort_limit = 8,

block_size = 64,

cacheline_size = 64

};

template<class Unsigned>
Unsigned log2(Unsigned n) {
Unsigned log = 0;
while (n >>= 1) ++log;
return log;
}

template<class Iter, class Compare>
inline bool partial_insertion_sort(Iter begin, Iter end, Compare comp) {
typedef typename boost::movelib::iterator_traits<Iter>::value_type T;
typedef typename boost::movelib::iterator_traits<Iter>::size_type  size_type;
if (begin == end) return true;

size_type limit = 0;
for (Iter cur = begin + 1; cur != end; ++cur) {
if (limit > partial_insertion_sort_limit) return false;

Iter sift = cur;
Iter sift_1 = cur - 1;

if (comp(*sift, *sift_1)) {
T tmp = boost::move(*sift);

do { *sift-- = boost::move(*sift_1); }
while (sift != begin && comp(tmp, *--sift_1));

*sift = boost::move(tmp);
limit += size_type(cur - sift);
}
}

return true;
}

template<class Iter, class Compare>
inline void sort2(Iter a, Iter b, Compare comp) {
if (comp(*b, *a)) boost::adl_move_iter_swap(a, b);
}

template<class Iter, class Compare>
inline void sort3(Iter a, Iter b, Iter c, Compare comp) {
sort2(a, b, comp);
sort2(b, c, comp);
sort2(a, b, comp);
}

template<class Iter, class Compare>
pdqsort_detail::pair<Iter, bool> partition_right(Iter begin, Iter end, Compare comp) {
typedef typename boost::movelib::iterator_traits<Iter>::value_type T;

T pivot(boost::move(*begin));

Iter first = begin;
Iter last = end;

while (comp(*++first, pivot));

if (first - 1 == begin) while (first < last && !comp(*--last, pivot));
else                    while (                !comp(*--last, pivot));

bool already_partitioned = first >= last;

while (first < last) {
boost::adl_move_iter_swap(first, last);
while (comp(*++first, pivot));
while (!comp(*--last, pivot));
}

Iter pivot_pos = first - 1;
*begin = boost::move(*pivot_pos);
*pivot_pos = boost::move(pivot);

return pdqsort_detail::pair<Iter, bool>(pivot_pos, already_partitioned);
}

template<class Iter, class Compare>
inline Iter partition_left(Iter begin, Iter end, Compare comp) {
typedef typename boost::movelib::iterator_traits<Iter>::value_type T;

T pivot(boost::move(*begin));
Iter first = begin;
Iter last = end;

while (comp(pivot, *--last));

if (last + 1 == end) while (first < last && !comp(pivot, *++first));
else                 while (                !comp(pivot, *++first));

while (first < last) {
boost::adl_move_iter_swap(first, last);
while (comp(pivot, *--last));
while (!comp(pivot, *++first));
}

Iter pivot_pos = last;
*begin = boost::move(*pivot_pos);
*pivot_pos = boost::move(pivot);

return pivot_pos;
}


template<class Iter, class Compare>
void pdqsort_loop( Iter begin, Iter end, Compare comp
, typename boost::movelib::iterator_traits<Iter>::size_type bad_allowed
, bool leftmost = true)
{
typedef typename boost::movelib::iterator_traits<Iter>::size_type size_type;

while (true) {
size_type size = size_type(end - begin);

if (size < insertion_sort_threshold) {
insertion_sort(begin, end, comp);
return;
}

size_type s2 = size / 2;
if (size > ninther_threshold) {
sort3(begin, begin + s2, end - 1, comp);
sort3(begin + 1, begin + (s2 - 1), end - 2, comp);
sort3(begin + 2, begin + (s2 + 1), end - 3, comp);
sort3(begin + (s2 - 1), begin + s2, begin + (s2 + 1), comp);
boost::adl_move_iter_swap(begin, begin + s2);
} else sort3(begin + s2, begin, end - 1, comp);

if (!leftmost && !comp(*(begin - 1), *begin)) {
begin = partition_left(begin, end, comp) + 1;
continue;
}

pdqsort_detail::pair<Iter, bool> part_result = partition_right(begin, end, comp);
Iter pivot_pos = part_result.first;
bool already_partitioned = part_result.second;

size_type l_size = size_type(pivot_pos - begin);
size_type r_size = size_type(end - (pivot_pos + 1));
bool highly_unbalanced = l_size < size / 8 || r_size < size / 8;

if (highly_unbalanced) {
if (--bad_allowed == 0) {
boost::movelib::heap_sort(begin, end, comp);
return;
}

if (l_size >= insertion_sort_threshold) {
boost::adl_move_iter_swap(begin,             begin + l_size / 4);
boost::adl_move_iter_swap(pivot_pos - 1, pivot_pos - l_size / 4);

if (l_size > ninther_threshold) {
boost::adl_move_iter_swap(begin + 1,         begin + (l_size / 4 + 1));
boost::adl_move_iter_swap(begin + 2,         begin + (l_size / 4 + 2));
boost::adl_move_iter_swap(pivot_pos - 2, pivot_pos - (l_size / 4 + 1));
boost::adl_move_iter_swap(pivot_pos - 3, pivot_pos - (l_size / 4 + 2));
}
}

if (r_size >= insertion_sort_threshold) {
boost::adl_move_iter_swap(pivot_pos + 1, pivot_pos + (1 + r_size / 4));
boost::adl_move_iter_swap(end - 1,                   end - r_size / 4);

if (r_size > ninther_threshold) {
boost::adl_move_iter_swap(pivot_pos + 2, pivot_pos + (2 + r_size / 4));
boost::adl_move_iter_swap(pivot_pos + 3, pivot_pos + (3 + r_size / 4));
boost::adl_move_iter_swap(end - 2,             end - (1 + r_size / 4));
boost::adl_move_iter_swap(end - 3,             end - (2 + r_size / 4));
}
}
} else {
if (already_partitioned && partial_insertion_sort(begin, pivot_pos, comp)
&& partial_insertion_sort(pivot_pos + 1, end, comp)) return;
}

pdqsort_loop<Iter, Compare>(begin, pivot_pos, comp, bad_allowed, leftmost);
begin = pivot_pos + 1;
leftmost = false;
}
}
}


template<class Iter, class Compare>
void pdqsort(Iter begin, Iter end, Compare comp)
{
if (begin == end) return;
typedef typename boost::movelib::iterator_traits<Iter>::size_type size_type;
pdqsort_detail::pdqsort_loop<Iter, Compare>(begin, end, comp, pdqsort_detail::log2(size_type(end - begin)));
}

}  
}  

#include <boost/move/detail/config_end.hpp>

#endif   
