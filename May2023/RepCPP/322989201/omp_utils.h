#ifndef FREE_TENSOR_OMP_UTILS_H
#define FREE_TENSOR_OMP_UTILS_H

#include <atomic>
#include <concepts>
#include <exception>
#include <functional>

#include <omp.h>

namespace freetensor {


template <std::integral T>
void exceptSafeParallelFor(T begin, T end, T step,
const std::function<void(T)> &body,
omp_sched_t schedKind,
int schedChunkSize = 0 ) {
std::atomic_flag hasExcept_ = ATOMIC_FLAG_INIT;

std::exception_ptr except_;

omp_set_schedule(schedKind, schedChunkSize);
#pragma omp parallel
{
#pragma omp for schedule(runtime)
for (auto i = begin; i < end; i += step) {
try {
body(i);
} catch (...) {
if (!hasExcept_.test_and_set()) {
except_ = std::current_exception();
#pragma omp cancel for
}
}
#pragma omp cancellation point for 
}
}
if (except_) {
std::rethrow_exception(except_);
}
}

} 

#endif 
