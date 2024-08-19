#pragma once

#include <CL/sycl.hpp>

#define CUDART_INF_F  0x7f800000

namespace facebook { namespace cuda {

template <typename T>
struct NumericLimits {};

template<>
struct NumericLimits<float> {
inline static float minPossible() {
return -sycl::bit_cast<float>(CUDART_INF_F);
}

inline static float maxPossible() {
return sycl::bit_cast<float>(CUDART_INF_F);
}
};

template<>
struct NumericLimits<int> {
inline static int minPossible() {
return INT_MIN;
}

inline static int maxPossible() {
return INT_MAX;
}
};

template<>
struct NumericLimits<unsigned int> {
inline static unsigned int minPossible() {
return 0;
}

inline static unsigned int maxPossible() {
return UINT_MAX;
}
};

} } 
