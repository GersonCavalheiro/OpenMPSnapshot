#pragma once
#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif
#ifdef __cplusplus
extern "C" {
#include <math.h>
#include <stdint.h>
#else
#include <stdbool.h>
#include <stdint.h>
#endif
typedef struct __attribute__((packed)) {
uint8_t r;
uint8_t g;
uint8_t b;
uint8_t a;
} rgba;
typedef uint8_t grayscale;
static inline bool rgba_isequal(const rgba * a, const rgba * b) {
return (a->r == b->r) &&
(a->g == b->g) &&
(a->b == b->b) &&
(a->a == b->a);
}
static DEVICE_CALLABLE rgba color1(float v) {
rgba c = {255U, 255U, 255U, 255U}; 
if (v < 0.25f) {
c.r = 0;
c.g = (char) rintf(1020.0f * v);
} else if (v < 0.5f) {
c.r = 0;
c.b = (char) rintf(255.0f * (1.0f + 4.0f * (0.25f - v)));
} else if (v < 0.75f) {
c.r = (char) rintf(1020.0f * (v - 0.5f));
c.b = 0;
} else {
c.g = (char) rintf(255.0f * (1.0f + 4.0f * (0.75f - v)));
c.b = 0;
}
return c;
}
static DEVICE_CALLABLE rgba color2(float x, float y) {
rgba c;
c.r = (unsigned char) rintf(255.0f * x);
c.g = (unsigned char) rintf(255.0f * y);
c.b = (unsigned char) rintf(127.5f * x + 127.5f * y);
c.a = 255U;
return c;
}
static DEVICE_CALLABLE rgba rgba_scramble(rgba orig, uint x, uint y) {
rgba result;
unsigned char * orig_as_vector = (unsigned char *) &orig;
unsigned char * result_as_vector = (unsigned char *) &result;
for (uint i = 0; i < 3; ++i) {
result_as_vector[(x + i) % 3] = orig_as_vector[i];
}
result.a = orig.a;
if (y % 8 == 0) {
} else if (y % 8 == 1) {
result.r = 255u - result.r;
} else if (y % 8 == 2) {
result.r = 255u - result.r;
result.g = 255u - result.g;
} else if (y % 8 == 3) {
result.r = 255u - result.r;
result.g = 255u - result.g;
result.b = 255u - result.b;
} else if (y % 8 == 4) {
result.g = 255u - result.g;
result.b = 255u - result.b;
} else if (y % 8 == 5) {
result.g = 255u - result.g;
} else if (y % 8 == 6) {
result.b = 255u - result.b;
} else if (y % 8 == 7) {
result.r = 255u - result.r;
result.b = 255u - result.b;
}
return result;
}
#ifdef __cplusplus
}
#endif
