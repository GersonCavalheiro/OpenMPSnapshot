#include <immintrin.h>
#include <stdio.h>
static __attribute__((always_inline)) inline void GatherRGBx8(const __m128i r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
const __m128i b7_g7_r7_b6_g6_r6_b5_g5,
__m128i *r7_r6_r5_r4_r3_r2_r1_r0,
__m128i *g7_g6_g5_g4_g3_g2_g1_g0,
__m128i *b7_b6_b5_b4_b3_b2_b1_b0)
{
const __m128i shuffle_mask = _mm_set_epi8(9, 6, 3, 0, 11, 8, 5, 2, 10, 7, 4, 1, 9, 6, 3, 0);
__m128i b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4 = _mm_alignr_epi8(b7_g7_r7_b6_g6_r6_b5_g5, r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0, 12);
__m128i r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0 = _mm_shuffle_epi8(r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0, shuffle_mask);
__m128i r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4 = _mm_shuffle_epi8(b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4, shuffle_mask);
__m128i b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4_r3_r2_r1_r0 = _mm_alignr_epi8(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4, r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0, 12);
__m128i g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz_zz_zz_zz_zz = _mm_slli_si128(r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0, 8);
__m128i zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4 = _mm_srli_si128(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4, 4);
__m128i r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_g3_g2_g1_g0 = _mm_alignr_epi8(zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4, g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz_zz_zz_zz_zz, 12);
__m128i b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz = _mm_slli_si128(r3_r2_r1_r0_b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0, 4);
__m128i zz_zz_zz_zz_zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4 = _mm_srli_si128(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4, 8);
__m128i zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_b3_b2_b1_b0 = _mm_alignr_epi8(zz_zz_zz_zz_zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4, b3_b2_b1_b0_g3_g2_g1_g0_r3_r2_r1_r0_zz_zz_zz_zz, 12);
*r7_r6_r5_r4_r3_r2_r1_r0 = _mm_cvtepu8_epi16(b7_b6_b5_b4_g7_g6_g5_g4_r7_r6_r5_r4_r3_r2_r1_r0);
*g7_g6_g5_g4_g3_g2_g1_g0 = _mm_cvtepu8_epi16(r7_r6_r5_r4_b7_b6_b5_b4_g7_g6_g5_g4_g3_g2_g1_g0);
*b7_b6_b5_b4_b3_b2_b1_b0 = _mm_cvtepu8_epi16(zz_zz_zz_zz_r7_r6_r5_r4_b7_b6_b5_b4_b3_b2_b1_b0);
}
static __attribute__((always_inline)) inline __m128i Rgb2Yx8(__m128i r7_r6_r5_r4_r3_r2_r1_r0,
__m128i g7_g6_g5_g4_g3_g2_g1_g0,
__m128i b7_b6_b5_b4_b3_b2_b1_b0)
{
const __m128i r_coef = _mm_set1_epi16((short)(0.2126 * 32768.0 + 0.5)); 
const __m128i g_coef = _mm_set1_epi16((short)(0.7152 * 32768.0 + 0.5)); 
const __m128i b_coef = _mm_set1_epi16((short)(0.0722 * 32768.0 + 0.5)); 
r7_r6_r5_r4_r3_r2_r1_r0 = _mm_slli_epi16(r7_r6_r5_r4_r3_r2_r1_r0, 6);
g7_g6_g5_g4_g3_g2_g1_g0 = _mm_slli_epi16(g7_g6_g5_g4_g3_g2_g1_g0, 6);
b7_b6_b5_b4_b3_b2_b1_b0 = _mm_slli_epi16(b7_b6_b5_b4_b3_b2_b1_b0, 6);
__m128i y7_y6_y5_y4_y3_y2_y1_y0 = _mm_add_epi16(_mm_add_epi16(
_mm_mulhrs_epi16(r7_r6_r5_r4_r3_r2_r1_r0, r_coef),
_mm_mulhrs_epi16(g7_g6_g5_g4_g3_g2_g1_g0, g_coef)),
_mm_mulhrs_epi16(b7_b6_b5_b4_b3_b2_b1_b0, b_coef));
y7_y6_y5_y4_y3_y2_y1_y0 = _mm_srli_epi16(y7_y6_y5_y4_y3_y2_y1_y0, 6);
return y7_y6_y5_y4_y3_y2_y1_y0;
}
void convert_memory_simd_sse(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
int floats_per_operation = 8;
int size = width * height;
int pixel_per_thread_unaligned = size / threads;
int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;
#pragma omp parallel for
for (int thread = 0; thread < threads; thread++)
{
int end;
if (thread + 1 == threads)
{
end = ((int)size / floats_per_operation) * floats_per_operation;
}
else
{
end = pixel_per_thread_aligned * (thread + 1);
}
__m128i r7_r6_r5_r4_r3_r2_r1_r0;
__m128i g7_g6_g5_g4_g3_g2_g1_g0;
__m128i b7_b6_b5_b4_b3_b2_b1_b0;
for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
{
__m128i r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0 = _mm_loadu_si128((__m128i *)&img[i * channels]); 
__m128i b7_g7_r7_b6_g6_r6_b5_g5 = _mm_loadu_si128((__m128i *)&img[i * channels + 16]);                    
GatherRGBx8(r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
b7_g7_r7_b6_g6_r6_b5_g5,
&r7_r6_r5_r4_r3_r2_r1_r0,
&g7_g6_g5_g4_g3_g2_g1_g0,
&b7_b6_b5_b4_b3_b2_b1_b0);
__m128i y7_y6_y5_y4_y3_y2_y1_y0 = Rgb2Yx8(r7_r6_r5_r4_r3_r2_r1_r0,
g7_g6_g5_g4_g3_g2_g1_g0,
b7_b6_b5_b4_b3_b2_b1_b0);
__m128i j7_j6_j5_j4_j3_j2_j1_j0 = _mm_packus_epi16(y7_y6_y5_y4_y3_y2_y1_y0, y7_y6_y5_y4_y3_y2_y1_y0);
_mm_storel_epi64((__m128i *)&result[i], j7_j6_j5_j4_j3_j2_j1_j0);
}
}
int start = ((int)size / floats_per_operation) * floats_per_operation;
for (int i = start; i < size; i++)
{
result[i] =
0.2126 * img[(i * channels)]        
+ 0.7152 * img[(i * channels) + 1]  
+ 0.0722 * img[(i * channels) + 2]; 
}
}
