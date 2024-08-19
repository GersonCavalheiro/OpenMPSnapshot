#ifndef STAN_MATH_OPENCL_KERNELS_DEVICE_ATOMIC_ADD_DOUBLE_HPP
#define STAN_MATH_OPENCL_KERNELS_DEVICE_ATOMIC_ADD_DOUBLE_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/stringify.hpp>
#include <string>

namespace stan {
namespace math {
namespace opencl_kernels {
static const char *atomic_add_double_device_function
= "\n"
"#ifndef STAN_MATH_OPENCL_KERNELS_DEVICE_FUNCTIONS_ATOMIC_ADD_DOUBLE\n"
"#define STAN_MATH_OPENCL_KERNELS_DEVICE_FUNCTIONS_ATOMIC_ADD_DOUBLE\n"
"#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable\n" STRINGIFY(

void atomic_add_double(__global double *val, double delta) {
union {
double f;
ulong i;
} old_val;
union {
double f;
ulong i;
} new_val;
do {
old_val.f = *val;
new_val.f = old_val.f + delta;
} while (atom_cmpxchg((volatile __global ulong *)val, old_val.i,
new_val.i)
!= old_val.i);
}

void local_atomic_add_double(__local double *val, double delta) {
union {
double f;
ulong i;
} old_val;
union {
double f;
ulong i;
} new_val;
do {
old_val.f = *val;
new_val.f = old_val.f + delta;
} while (atom_cmpxchg((volatile __local ulong *)val, old_val.i,
new_val.i)
!= old_val.i);
}
) "\n#endif\n";  

}  
}  
}  

#endif
#endif
