#pragma once

#include <CL/sycl.hpp>

namespace facebook { namespace cuda {


template <typename T>
struct GreaterThan {
static inline bool compare(const T lhs, const T rhs) {
return (lhs > rhs);
}
};

template <typename T>
struct LessThan {
static inline bool compare(const T lhs, const T rhs) {
return (lhs < rhs);
}
};

} } 
