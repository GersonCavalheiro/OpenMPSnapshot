#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2> void matanh_core(const F1 &A, F2 &C) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(A, C));
assert(util::is_same_structure(A, C));
assert(util::is_same_device_mem_stat(A, C));

internal::vatanh(A.get_nnz(), A.data(), C.data(), A.get_device_mem_stat());

logger.func_out();
}
} 
} 
