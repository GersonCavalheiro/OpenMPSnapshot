#pragma once

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3>
void stadd_core(const F1 &A, const F2 alpha, F3 &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(A, C));
assert(util::is_same_device_mem_stat(A, C));

internal::vadd(A.get_nnz(), A.data(), alpha, C.data(),
C.get_device_mem_stat());

logger.func_out();
}

template <typename F1, typename F2, typename F3>
void ttadd_core(const F1 &A, const F2 &B, F3 &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(A, B, C));
assert(util::is_same_structure(A, B, C));
assert(util::is_same_device_mem_stat(A, B, C));

internal::vadd(A.get_nnz(), A.data(), B.data(), C.data(),
C.get_device_mem_stat());

logger.func_out();
}
} 

} 
