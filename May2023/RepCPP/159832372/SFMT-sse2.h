#pragma once


#ifndef SFMT_SSE2_H
#define SFMT_SSE2_H

inline static void mm_recursion(__m128i * r, __m128i a, __m128i b,
__m128i c, __m128i d);


inline static void mm_recursion(__m128i * r, __m128i a, __m128i b,
__m128i c, __m128i d)
{
__m128i v, x, y, z;

y = _mm_srli_epi32(b, SFMT_SR1);
z = _mm_srli_si128(c, SFMT_SR2);
v = _mm_slli_epi32(d, SFMT_SL1);
z = _mm_xor_si128(z, a);
z = _mm_xor_si128(z, v);
x = _mm_slli_si128(a, SFMT_SL2);
y = _mm_and_si128(y, sse2_param_mask.si);
z = _mm_xor_si128(z, x);
z = _mm_xor_si128(z, y);
*r = z;
}


void sfmt_gen_rand_all(sfmt_t * sfmt) {
int i;
__m128i r1, r2;
w128_t * pstate = sfmt->state;

r1 = pstate[SFMT_N - 2].si;
r2 = pstate[SFMT_N - 1].si;
for (i = 0; i < SFMT_N - SFMT_POS1; i++) {
mm_recursion(&pstate[i].si, pstate[i].si,
pstate[i + SFMT_POS1].si, r1, r2);
r1 = r2;
r2 = pstate[i].si;
}
for (; i < SFMT_N; i++) {
mm_recursion(&pstate[i].si, pstate[i].si,
pstate[i + SFMT_POS1 - SFMT_N].si,
r1, r2);
r1 = r2;
r2 = pstate[i].si;
}
}


static void gen_rand_array(sfmt_t * sfmt, w128_t * array, int size)
{
int i, j;
__m128i r1, r2;
w128_t * pstate = sfmt->state;

r1 = pstate[SFMT_N - 2].si;
r2 = pstate[SFMT_N - 1].si;
for (i = 0; i < SFMT_N - SFMT_POS1; i++) {
mm_recursion(&array[i].si, pstate[i].si,
pstate[i + SFMT_POS1].si, r1, r2);
r1 = r2;
r2 = array[i].si;
}
for (; i < SFMT_N; i++) {
mm_recursion(&array[i].si, pstate[i].si,
array[i + SFMT_POS1 - SFMT_N].si, r1, r2);
r1 = r2;
r2 = array[i].si;
}
for (; i < size - SFMT_N; i++) {
mm_recursion(&array[i].si, array[i - SFMT_N].si,
array[i + SFMT_POS1 - SFMT_N].si, r1, r2);
r1 = r2;
r2 = array[i].si;
}
for (j = 0; j < 2 * SFMT_N - size; j++) {
pstate[j] = array[j + size - SFMT_N];
}
for (; i < size; i++, j++) {
mm_recursion(&array[i].si, array[i - SFMT_N].si,
array[i + SFMT_POS1 - SFMT_N].si, r1, r2);
r1 = r2;
r2 = array[i].si;
pstate[j] = array[i];
}
}


#endif
