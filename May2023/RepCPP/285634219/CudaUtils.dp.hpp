#pragma once

#include <CL/sycl.hpp>

namespace facebook { namespace cuda {


template <typename T> inline constexpr T ceil(T a, T b) {
return (a + b - 1) / b;
}


template <typename T> inline constexpr T floor(T a, T b) {
return (a - b + 1) / b;
}


inline int getWarpId(sycl::nd_item<1> &item) {
return item.get_local_id(0) / WARP_SIZE;
}


inline int getThreadsInBlock(sycl::nd_item<1> &item) {
return item.get_local_range(0);
}


inline int getWarpsInBlock(sycl::nd_item<1> &item) {
return ceil(getThreadsInBlock(item), WARP_SIZE);
}



inline int getLaneId(sycl::nd_item<1> &item) {
int laneId = item.get_local_id(0) % WARP_SIZE;
return laneId;
}



inline int getBit(int val, int pos) {
return (val >> pos) & 0x1;
}



inline constexpr int getMSB(int val) {
return
((val >= 1024 && val < 2048) ? 10 :
((val >= 512) ? 9 :
((val >= 256) ? 8 :
((val >= 128) ? 7 :
((val >= 64) ? 6 :
((val >= 32) ? 5 :
((val >= 16) ? 4 :
((val >= 8) ? 3 :
((val >= 4) ? 2 :
((val >= 2) ? 1 :
((val == 1) ? 0 : -1)))))))))));
}

} }  
