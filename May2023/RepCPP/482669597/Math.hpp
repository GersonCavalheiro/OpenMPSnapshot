#ifndef __MATH_HPP__
#define __MATH_HPP__

#include <omp.h>

namespace Impl {
template <class ViewType>
void swap(ViewType &a, ViewType &b) {
ViewType tmp = a;
a = b;
b = tmp;
}





void deep_copy(RealView4D &a, const RealView4D &b) {
shape_nd<4> shape = a.extents();
range_nd<4> offset = a.offsets();

int n0 = shape[0], n1 = shape[1], n2 = shape[2], n3 = shape[3];
int n0_start = offset[0], n1_start = offset[1], n2_start = offset[2], n3_start = offset[3];
int n0_end   = n0 + n0_start, n1_end = n1 + n1_start, n2_end = n2 + n2_start, n3_end = n3 + n3_start;

#if defined( ENABLE_OPENMP_OFFLOAD )
#pragma omp target teams distribute parallel for simd collapse(4)
#else
#pragma omp parallel for collapse(2)
#endif
for(int jvy = n3_start; jvy < n3_end; jvy++) {
for(int jvx = n2_start; jvx < n2_end; jvx++) {
for(int jy = n1_start; jy < n1_end; jy++) {
for(int jx = n0_start; jx < n0_end; jx++) {
a(jx, jy, jvx, jvy) = b(jx, jy, jvx, jvy);
}
}
}
}
}
};

#endif
