#ifndef UPDATES_CUH
#define UPDATES_CUH

#include "helper/helpers.h"
#include "helper/rngpu.h"

inline HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask_many(const uint8_t factorDim, fast_kiss_state32_t state, const uint32_t rand_depth) {
uint32_t bit_flip_mask = FULLMASK >> (32-factorDim);
#pragma unroll
for(int i = 0; i < rand_depth; ++i) {
bit_flip_mask &= fast_kiss32(state);
}
return bit_flip_mask;
}

inline HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask_all(const uint8_t factorDim) {
return FULLMASK >> (32-factorDim);
}

inline HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask_one(const uint8_t factorDim, fast_kiss_state32_t state) {
const uint32_t lane = fast_kiss32(state) % factorDim;
return 1 << lane;
}

inline HOST_DEVICE_QUALIFIER
uint32_t get_flip_mask(const uint8_t factorDim, fast_kiss_state32_t state,
const float flipManyChance,
const uint32_t flipManyDepth) {
const float random_many = fast_kiss32(state) / (float) UINT32_MAX;

return random_many < flipManyChance ? get_flip_mask_many(factorDim, state, flipManyDepth)
: get_flip_mask_one(factorDim, state);
}

template<typename error_t>
inline HOST_DEVICE_QUALIFIER
bool metro(fast_kiss_state32_t state, const error_t error, const float temperature, const int error_max = 1) {
if(error <= 0) return true;
if(temperature <= 0) return false;
const float randomNumber = fast_kiss32(state) / (float) UINT32_MAX;
const float metro = sycl::exp((float)-error / error_max / temperature);
return randomNumber < metro;
}


template<typename error_t>
inline HOST_DEVICE_QUALIFIER
error_t error_measure(const int test, const int truth, const error_t weigth) {
return (truth == 1) ? weigth * (test ^ truth) : (test ^ truth);
}

#endif
