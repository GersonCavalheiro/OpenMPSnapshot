#pragma once
#include <cstdint>





class xorshift32_engine {
uint32_t state;

public:
xorshift32_engine(uint32_t state)
: state(state) {}

uint32_t operator()() {
uint32_t x = state;
x ^= x << 13;
x ^= x >> 17;
x ^= x << 5;
return state = x;
}
};
