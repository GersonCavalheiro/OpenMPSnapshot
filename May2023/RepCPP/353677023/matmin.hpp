#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void mmmin_core(const F1 &A, const F2 &B, F3 &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(A, B, C));
assert(util::is_same_structure(A, B, C));
assert(util::is_same_device_mem_stat(A, B, C));

internal::vmin(C.get_nnz(), A.data(), B.data(), C.data(),
C.get_device_mem_stat());

logger.func_out();
}

template <typename F1, typename F2, typename F3>
void smmin_core(const F1 &A, const F2 alpha, F3 &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(A, C));
assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, C));

internal::vmin(C.get_nnz(), A.data(), alpha, C.data(),
C.get_device_mem_stat());

logger.func_out();
}

template <typename F1, typename F2> F2 mmin_core(const F1 &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

F2 min = internal::vmin(C.get_nnz(), C.data(), C.get_device_mem_stat());

logger.func_out();
if (typeid(F1) == typeid(matrix::Dense<F2>)) {
return min;
} else {
return std::min(min, (F2)0.0);
}
}
} 
} 
