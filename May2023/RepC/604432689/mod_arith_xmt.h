#ifndef MOD_ARITH_XMT_H
#define MOD_ARITH_XMT_H
#include <stdint.h>
#include <assert.h>
#include <machine/mtaops.h>
#pragma mta inline
static inline uint_fast32_t mod_add(uint_fast32_t a, uint_fast32_t b) {
uint_fast32_t x;
assert (a <= 0x7FFFFFFE);
assert (b <= 0x7FFFFFFE);
x = a + b; 
return (x + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, x)) & 0x7FFFFFFF;
}
#pragma mta inline
static inline uint_fast32_t mod_mul(uint_fast32_t a, uint_fast32_t b) {
uint_fast64_t temp;
uint_fast32_t temp2;
assert (a <= 0x7FFFFFFE);
assert (b <= 0x7FFFFFFE);
temp = MTA_INT_ADD_MUL(0, a, b);
return (temp + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, temp)) & 0x7FFFFFFF;
}
#pragma mta inline
static inline uint_fast32_t mod_mac(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b) {
uint_fast64_t temp;
uint_fast32_t temp2;
assert (sum <= 0x7FFFFFFE);
assert (a <= 0x7FFFFFFE);
assert (b <= 0x7FFFFFFE);
temp = MTA_INT_ADD_MUL(sum, a, b);
return (temp + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, temp)) & 0x7FFFFFFF;
}
#pragma mta inline
static inline uint_fast32_t mod_mac2(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b, uint_fast32_t c, uint_fast32_t d) {
uint_fast64_t temp;
assert (sum <= 0x7FFFFFFE);
assert (a <= 0x7FFFFFFE);
assert (b <= 0x7FFFFFFE);
assert (c <= 0x7FFFFFFE);
assert (d <= 0x7FFFFFFE);
temp = MTA_INT_ADD_MUL(MTA_INT_ADD_MUL(sum, a, b), c, d);
return (temp + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, temp)) & 0x7FFFFFFF;
}
#pragma mta inline
static inline uint_fast32_t mod_mac3(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b, uint_fast32_t c, uint_fast32_t d, uint_fast32_t e, uint_fast32_t f) {
uint_fast64_t temp;
assert (sum <= 0x7FFFFFFE);
assert (a <= 0x7FFFFFFE);
assert (b <= 0x7FFFFFFE);
assert (c <= 0x7FFFFFFE);
assert (d <= 0x7FFFFFFE);
assert (e <= 0x7FFFFFFE);
assert (f <= 0x7FFFFFFE);
temp = MTA_INT_ADD_MUL(MTA_INT_ADD_MUL(MTA_INT_ADD_MUL(sum, a, b), c, d), e, f);
return (temp + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, temp)) & 0x7FFFFFFF;
}
#pragma mta inline
static inline uint_fast32_t mod_mac4(uint_fast32_t sum, uint_fast32_t a, uint_fast32_t b, uint_fast32_t c, uint_fast32_t d, uint_fast32_t e, uint_fast32_t f, uint_fast32_t g, uint_fast32_t h) {
uint_fast64_t temp;
assert (sum <= 0x7FFFFFFE);
assert (a <= 0x7FFFFFFE);
assert (b <= 0x7FFFFFFE);
assert (c <= 0x7FFFFFFE);
assert (d <= 0x7FFFFFFE);
assert (e <= 0x7FFFFFFE);
assert (f <= 0x7FFFFFFE);
assert (g <= 0x7FFFFFFE);
assert (h <= 0x7FFFFFFE);
temp = MTA_INT_ADD_MUL(MTA_INT_ADD_MUL(MTA_INT_ADD_MUL(MTA_INT_ADD_MUL(sum, a, b), c, d), e, f), g, h);
return (temp + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, temp)) & 0x7FFFFFFF;
}
static inline uint_fast64_t mod_down(uint_fast64_t x);
#pragma mta inline
static inline uint_fast64_t mod_down_fast(uint_fast64_t x) {
return (x >> 31) + (x & 0x7FFFFFFF);
}
#pragma mta inline
static inline uint_fast64_t mod_down(uint_fast64_t x) {
return ((x + MTA_UNS_ADD_MUL_UPPER(0x0000000200000003, 0x0000000200000004, x)) & 0x7FFFFFFF);
}
#pragma mta inline
static inline uint_fast32_t mod_mul_x(uint_fast32_t a) {
return mod_mul(a, 107374182);
}
#pragma mta inline
static inline uint_fast32_t mod_mul_y(uint_fast32_t a) {
return mod_mul(a, 104480);
}
#pragma mta inline
static inline uint_fast32_t mod_mac_y(uint_fast32_t sum, uint_fast32_t a) {
return mod_mac(sum, a, 104480);
}
#endif 
