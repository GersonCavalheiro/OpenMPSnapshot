#ifndef PSS_HPP
#define PSS_HPP



#include "pss_common.hpp"

namespace pss {

namespace internal {

template <typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename RandomAccessIterator3,
typename Compare>
void parallel_move_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe,
RandomAccessIterator2 ys, RandomAccessIterator2 ye,
RandomAccessIterator3 zs,
bool destroy, Compare comp,
ssize_t cutoff) {
while( (xe-xs) + (ye-ys) > cutoff ) {
RandomAccessIterator1 xm;
RandomAccessIterator2 ym;
if( xe-xs < ye-ys  ) {
ym = ys+(ye-ys)/2;
xm = std::upper_bound(xs,xe,*ym,comp);
} else {
xm = xs+(xe-xs)/2;
ym = std::lower_bound(ys,ye,*xm,comp);
}
#pragma omp task untied mergeable firstprivate(xs,xm,ys,ym,zs,destroy,comp)
parallel_move_merge( xs, xm, ys, ym, zs, destroy, comp, cutoff );
zs += (xm-xs) + (ym-ys);
xs = xm;
ys = ym;
}
serial_move_merge( xs, xe, ys, ye, zs, comp );
if( destroy ) {
serial_destroy( xs, xe );
serial_destroy( ys, ye );
}
#pragma omp taskwait
}

template <typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename Compare>
void parallel_stable_sort_aux(RandomAccessIterator1 xs, RandomAccessIterator1 xe,
RandomAccessIterator2 zs,
int inplace, Compare comp,
ssize_t cutoff) {
if((xe - xs) <= cutoff) {
stable_sort_base_case(xs, xe, zs, inplace, comp);
} else {
RandomAccessIterator1 xm = xs + (xe-xs)/2;
RandomAccessIterator2 zm = zs + (xm-xs);
RandomAccessIterator2 ze = zs + (xe-xs);
#pragma omp task
parallel_stable_sort_aux( xs, xm, zs, !inplace, comp, cutoff );
parallel_stable_sort_aux( xm, xe, zm, !inplace, comp, cutoff );
#pragma omp taskwait
if( inplace )
parallel_move_merge( zs, zm, zm, ze, xs, inplace==2, comp, cutoff );
else
parallel_move_merge( xs, xm, xm, xe, zs, false, comp, cutoff );
}
}

} 

template<typename RandomAccessIterator, typename Compare>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe,
Compare comp) {
auto n = xe - xs;
auto t = omp_get_max_threads();
auto cutoff = n / t;
if (cutoff < 2) cutoff = 2;
typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
internal::raw_buffer z(size_t(n) * sizeof(T));
#pragma omp parallel
#pragma omp master
internal::parallel_stable_sort_aux( xs, xe, static_cast<T*>(z.get()), 2, comp, cutoff );
}

} 

#endif
