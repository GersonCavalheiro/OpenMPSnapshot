#pragma once

#include <cstdint>

namespace quids::utils {
class random_generator {
private:
uint64_t shuffle_table[2];
public:
random_generator() {
shuffle_table[0] = rand();
shuffle_table[1] = rand();
}

float operator()() {
uint64_t s1 = shuffle_table[0];
uint64_t s0 = shuffle_table[1];
uint64_t result = s0 + s1;
shuffle_table[0] = s0;
s1 ^= s1 << 23;
shuffle_table[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
return (float)result / (float)((uint64_t)0xffffffff);
}
};
}