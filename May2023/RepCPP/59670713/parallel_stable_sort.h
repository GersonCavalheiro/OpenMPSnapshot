#pragma once


#include <omp.h>
#include <algorithm>
#include <utility>
#include <iterator>

namespace pss {

namespace internal {

template <class RandomAccessIterator>
void serial_destroy(RandomAccessIterator zs, RandomAccessIterator ze) {
typedef typename ::std::iterator_traits<RandomAccessIterator>::value_type T;
while (zs != ze) {
--ze;
(*ze).~T();
}
}

template <class RandomAccessIterator1, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
void serial_move_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 ys, RandomAccessIterator2 ye, RandomAccessIterator3 zs,
Compare comp) {
if (xs != xe) {
if (ys != ye) {
for (;;) {
if (comp(*ys, *xs)) {
*zs = ::std::move(*ys);
++zs;
if (++ys == ye) {
break;
}
} else {
*zs = ::std::move(*xs);
++zs;
if (++xs == xe) {
goto movey;
}
}
}
}
ys = xs;
ye = xe;
}
movey:
::std::move(ys, ye, zs);
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void stable_sort_base_case(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 zs, int inplace, Compare comp) {
::std::stable_sort(xs, xe, comp);
if (inplace != 2) {
RandomAccessIterator2 ze = zs + (xe - xs);
typedef typename ::std::iterator_traits<RandomAccessIterator2>::value_type T;
if (inplace) {
for (; zs < ze; ++zs) {
new (&*zs) T;
}
} else {
for (; zs < ze; ++xs, ++zs) {
new (&*zs) T(::std::move(*xs));
}
}
}
}

class raw_buffer {
void* ptr;

public:
raw_buffer(size_t bytes) : ptr(operator new(bytes, ::std::nothrow)) {
}

raw_buffer(const raw_buffer&) = delete;

raw_buffer& operator=(const raw_buffer&) = delete;

operator bool() const {
return ptr;
}
void* get() const {
return ptr;
}
~raw_buffer() {
operator delete(ptr);
}
};

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename Compare>
void parallel_move_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 ys, RandomAccessIterator2 ye, RandomAccessIterator3 zs,
bool destroy, Compare comp) {
const size_t MERGE_CUT_OFF = 2000;
while (static_cast<size_t>((xe - xs) + (ye - ys)) > MERGE_CUT_OFF) {
RandomAccessIterator1 xm;
RandomAccessIterator2 ym;
if (xe - xs < ye - ys) {
ym = ys + (ye - ys) / 2;
xm = ::std::upper_bound(xs, xe, *ym, comp);
} else {
xm = xs + (xe - xs) / 2;
ym = ::std::lower_bound(ys, ye, *xm, comp);
}
#pragma omp task untied mergeable firstprivate(xs, xm, ys, ym, zs, destroy, comp)
parallel_move_merge(xs, xm, ys, ym, zs, destroy, comp);
zs += (xm - xs) + (ym - ys);
xs = xm;
ys = ym;
}
serial_move_merge(xs, xe, ys, ye, zs, comp);
if (destroy) {
serial_destroy(xs, xe);
serial_destroy(ys, ye);
}
#pragma omp taskwait
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
void parallel_stable_sort_aux(RandomAccessIterator1 xs, RandomAccessIterator1 xe, RandomAccessIterator2 zs, int inplace, Compare comp) {
const size_t SORT_CUT_OFF = 500;
if (static_cast<size_t>(xe - xs) <= SORT_CUT_OFF) {
stable_sort_base_case(xs, xe, zs, inplace, comp);
} else {
RandomAccessIterator1 xm = xs + (xe - xs) / 2;
RandomAccessIterator2 zm = zs + (xm - xs);
RandomAccessIterator2 ze = zs + (xe - xs);
#pragma omp task
parallel_stable_sort_aux(xs, xm, zs, !inplace, comp);
parallel_stable_sort_aux(xm, xe, zm, !inplace, comp);
#pragma omp taskwait
if (inplace) {
parallel_move_merge(zs, zm, zm, ze, xs, inplace == 2, comp);
} else {
parallel_move_merge(xs, xm, xm, xe, zs, false, comp);
}
}
}

}  

template <typename RandomAccessIterator, typename Compare>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
typedef typename ::std::iterator_traits<RandomAccessIterator>::value_type T;
internal::raw_buffer z(sizeof(T) * (xe - xs));
if (z) {
if (omp_get_num_threads() > 1) {
internal::parallel_stable_sort_aux(xs, xe, static_cast<T*>(z.get()), 2, comp);
} else {
#pragma omp parallel
#pragma omp master
internal::parallel_stable_sort_aux(xs, xe, static_cast<T*>(z.get()), 2, comp);
}
} else {
::std::stable_sort(xs, xe, comp);
}
}

template <class RandomAccessIterator>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe) {
typedef typename ::std::iterator_traits<RandomAccessIterator>::value_type T;
parallel_stable_sort(xs, xe, ::std::less<T>());
}

}  
