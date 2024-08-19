#pragma once

#include <stddef.h>
#include <stdint.h>

struct Chacha20Block {

uint32_t state[16];

static uint32_t rotl32(uint32_t x, int n) {
return (x << n) | (x >> (32 - n));
}

static uint32_t pack4(const uint8_t *a) {
return
uint32_t(a[0] << 0*8) |
uint32_t(a[1] << 1*8) |
uint32_t(a[2] << 2*8) |
uint32_t(a[3] << 3*8);
}

static void unpack4(uint32_t src, uint8_t *dst) {
dst[0] = (src >> 0*8) & 0xff;
dst[1] = (src >> 1*8) & 0xff;
dst[2] = (src >> 2*8) & 0xff;
dst[3] = (src >> 3*8) & 0xff;
}

Chacha20Block(const uint8_t key[32], const uint8_t nonce[8]) {
const uint8_t *magic_constant = (uint8_t*)"expand 32-byte k";
state[ 0] = pack4(magic_constant + 0*4);
state[ 1] = pack4(magic_constant + 1*4);
state[ 2] = pack4(magic_constant + 2*4);
state[ 3] = pack4(magic_constant + 3*4);
state[ 4] = pack4(key + 0*4);
state[ 5] = pack4(key + 1*4);
state[ 6] = pack4(key + 2*4);
state[ 7] = pack4(key + 3*4);
state[ 8] = pack4(key + 4*4);
state[ 9] = pack4(key + 5*4);
state[10] = pack4(key + 6*4);
state[11] = pack4(key + 7*4);
state[12] = 0;
state[13] = 0;
state[14] = pack4(nonce + 0*4);
state[15] = pack4(nonce + 1*4);
}

void set_counter(uint64_t counter) {
state[12] = uint32_t(counter);
state[13] = counter >> 32;
}

void next(uint32_t result[16]) {
for (int i = 0; i < 16; i++) result[i] = state[i];

#define CHACHA20_QUARTERROUND(x, a, b, c, d) \
x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); \
x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8); \
x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

for (int i = 0; i < 10; i++) {
CHACHA20_QUARTERROUND(result, 0, 4, 8, 12)
CHACHA20_QUARTERROUND(result, 1, 5, 9, 13)
CHACHA20_QUARTERROUND(result, 2, 6, 10, 14)
CHACHA20_QUARTERROUND(result, 3, 7, 11, 15)
CHACHA20_QUARTERROUND(result, 0, 5, 10, 15)
CHACHA20_QUARTERROUND(result, 1, 6, 11, 12)
CHACHA20_QUARTERROUND(result, 2, 7, 8, 13)
CHACHA20_QUARTERROUND(result, 3, 4, 9, 14)
}

for (int i = 0; i < 16; i++) result[i] += state[i];

uint32_t *counter = state + 12;
counter[0]++;
if (0 == counter[0]) {
counter[1]++;
}
}

void next(uint8_t result8[64]) {
uint32_t temp32[16];

next(temp32);

for (size_t i = 0; i < 16; i++) unpack4(temp32[i], result8 + i*4);
}
};

struct Chacha20 {

Chacha20Block block;
uint8_t keystream8[64];
size_t position;

Chacha20(const uint8_t key[32], const uint8_t nonce[8], uint64_t counter = 0):
block(key, nonce), position(64) {
block.set_counter(counter);
}

void crypt(uint8_t *bytes, size_t n_bytes) {
for (size_t i = 0; i < n_bytes; i++) {
if (position >= 64) {
block.next(keystream8);
position = 0;
}
bytes[i] ^= keystream8[position];
position++;
}
}
};
