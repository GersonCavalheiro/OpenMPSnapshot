#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2> void vsqrt_core(const F1 &a, F2 &y) {
Logger &logger = Logger::get_instance();
logger.func_in(monolish_func);

assert(util::is_same_size(a, y));
assert(util::is_same_device_mem_stat(a, y));

internal::vsqrt(y.size(), a.data() + a.get_offset(),
y.data() + y.get_offset(), y.get_device_mem_stat());

logger.func_out();
}
} 

} 
