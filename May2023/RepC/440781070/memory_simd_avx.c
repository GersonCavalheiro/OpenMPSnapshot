#include <immintrin.h>
#include <stdio.h>
static __attribute__((always_inline)) inline void GatherRGBx8_avx(
const __m256i gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
const __m256i bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
__m256i *rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
__m256i *gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
__m256i *bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0)
{
const __m256i brg_gbr_shuffle_mask =
_mm256_set_epi8(13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3, 0,  13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3, 0);
const __m256i grb_shuffle_mask =
_mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  14, 11, 8, 5, 2, 13, 10, 7, 4, 1, 15, 12, 9, 6, 3, 0);
const __m256i low_blend_mask =
_mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128);
const __m256i middle_blend_mask =
_mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0);
const __m256i high_blend_mask =
_mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
const __m256i zero_vec = _mm256_setzero_si256();
__m256i b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0 =
_mm256_shuffle_epi8(
gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
brg_gbr_shuffle_mask);
__m256i g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5 =
_mm256_permute2x128_si256(
b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
1);
__m256i g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 =
_mm256_blendv_epi8(
g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5,
b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
low_blend_mask);
__m256i rF_rE_rD_rC_rB_gF_gE_gD_gC_gB_bF_bE_bD_bC_bB_bA =
_mm256_shuffle_epi8(
bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
brg_gbr_shuffle_mask);
__m256i r_8bit = _mm256_blendv_epi8(
g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
rF_rE_rD_rC_rB_gF_gE_gD_gC_gB_bF_bE_bD_bC_bB_bA,
high_blend_mask);
__m256i g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0 =
_mm256_blendv_epi8(
g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
middle_blend_mask);
__m256i b_8bit = _mm256_alignr_epi8(
rF_rE_rD_rC_rB_gF_gE_gD_gC_gB_bF_bE_bD_bC_bB_bA,
g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
6);
__m256i xx_xx_xx_xx_xx_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 =
_mm256_alignr_epi8(
g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0_b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5,
b9_b8_b7_b6_b5_rA_r9_r8_r7_r6_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0_b4_b3_b2_b1_b0_r5_r4_r3_r2_r1_r0,
11);
__m256i gF_gE_gD_gC_gB_rF_rE_rD_rC_rB_bF_bE_bD_bC_bB_bA =
_mm256_shuffle_epi8(
bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
grb_shuffle_mask);
__m256i g_8bit = _mm256_blendv_epi8(
xx_xx_xx_xx_xx_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
gF_gE_gD_gC_gB_rF_rE_rD_rC_rB_bF_bE_bD_bC_bB_bA,
high_blend_mask);
__m256i xx_xx_xx_xx_xx_xx_xx_xx_rF_rE_rD_rC_rB_rA_r9_r8_xx_xx_xx_xx_xx_xx_xx_xx_r7_r6_r5_r4_r3_r2_r1_r0 =
_mm256_permute4x64_epi64(r_8bit, _MM_SHUFFLE(0, 1, 0, 0));
__m256i xx_xx_xx_xx_xx_xx_xx_xx_gF_gE_gD_gC_gB_gA_g9_g8_xx_xx_xx_xx_xx_xx_xx_xx_g7_g6_g5_g4_g3_g2_g1_g0 =
_mm256_permute4x64_epi64(g_8bit, _MM_SHUFFLE(0, 1, 0, 0));
__m256i xx_xx_xx_xx_xx_xx_xx_xx_bF_bE_bD_bC_bB_bA_b9_b8_xx_xx_xx_xx_xx_xx_xx_xx_b7_b6_b5_b4_b3_b2_b1_b0 =
_mm256_permute4x64_epi64(b_8bit, _MM_SHUFFLE(0, 1, 0, 0));
*rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 =
_mm256_unpacklo_epi8(
xx_xx_xx_xx_xx_xx_xx_xx_rF_rE_rD_rC_rB_rA_r9_r8_xx_xx_xx_xx_xx_xx_xx_xx_r7_r6_r5_r4_r3_r2_r1_r0,
zero_vec);
*gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 =
_mm256_unpacklo_epi8(
xx_xx_xx_xx_xx_xx_xx_xx_gF_gE_gD_gC_gB_gA_g9_g8_xx_xx_xx_xx_xx_xx_xx_xx_g7_g6_g5_g4_g3_g2_g1_g0,
zero_vec);
*bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0 =
_mm256_unpacklo_epi8(
xx_xx_xx_xx_xx_xx_xx_xx_bF_bE_bD_bC_bB_bA_b9_b8_xx_xx_xx_xx_xx_xx_xx_xx_b7_b6_b5_b4_b3_b2_b1_b0,
zero_vec);
}
static __attribute__((always_inline)) inline __m256i Rgb2Yx8_avx(__m256i rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
__m256i gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
__m256i bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0)
{
const __m256i r_coef = _mm256_set1_epi16((short)(6966.9768));  
const __m256i g_coef = _mm256_set1_epi16((short)(23436.1736)); 
const __m256i b_coef = _mm256_set1_epi16((short)(2366.3496));  
rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0 = _mm256_slli_epi16(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0, 6);
gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0 = _mm256_slli_epi16(gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0, 6);
bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0 = _mm256_slli_epi16(bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0, 6);
__m256i y7_y6_y5_y4_y3_y2_y1_y0 = _mm256_add_epi16(_mm256_add_epi16(
_mm256_mulhrs_epi16(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0, r_coef),
_mm256_mulhrs_epi16(gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0, g_coef)),
_mm256_mulhrs_epi16(bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0, b_coef));
return _mm256_srli_epi16(y7_y6_y5_y4_y3_y2_y1_y0, 6);
}
void convert_memory_simd_avx(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
const int pixel_per_iteration = 32;
const int size = width * height;
const int pixel_per_thread_unaligned = size / threads;
const int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / pixel_per_iteration) * pixel_per_iteration;
#pragma omp parallel for
for (int thread = 0; thread < threads; thread++)
{
int end;
if (thread + 1 == threads)
{
end = ((int)size / pixel_per_iteration) * pixel_per_iteration;
}
else
{
end = pixel_per_thread_aligned * (thread + 1);
}
__m256i rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0;
__m256i gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0;
__m256i bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0;
for (int i = pixel_per_thread_aligned * thread; i < end; i += pixel_per_iteration)
{
const int pixel_index = i * channels;
__m256i gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0 = _mm256_load_si256((__m256i *)&img[pixel_index]);
__m256i bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA = _mm256_load_si256((__m256i *)&img[pixel_index + 32]);
GatherRGBx8_avx(gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
&rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
&gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
&bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0);
__m256i y7_y6_y5_y4_y3_y2_y1_y0 = Rgb2Yx8_avx(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0);
gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0 = _mm256_loadu_si256((__m256i *)&img[pixel_index + 48]);
bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA = _mm256_loadu_si256((__m256i *)&img[pixel_index + 80]);
GatherRGBx8_avx(gA_rA_b9_g9_r9_b8_g8_r8_b7_g7_r7_b6_g6_r6_b5_g5_r5_b4_g4_r4_b3_g3_r3_b2_g2_r2_b1_g1_r1_b0_g0_r0,
bF_gF_rF_bE_gE_rE_bD_gD_rD_bC_gC_rC_bB_gB_rB_bA,
&rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
&gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
&bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0);
__m256i yF_yE_yD_yC_yB_yA_y9_y8 = Rgb2Yx8_avx(rF_rE_rD_rC_rB_rA_r9_r8_r7_r6_r5_r4_r3_r2_r1_r0,
gF_gE_gD_gC_gB_gA_g9_g8_g7_g6_g5_g4_g3_g2_g1_g0,
bF_bE_bD_bC_bB_bA_b9_b8_b7_b6_b5_b4_b3_b2_b1_b0);
__m256i yF_yE_yD_yC_y7_y6_y5_y4_yB_yA_y9_y8_y3_y2_y1_y0 = _mm256_packus_epi16(y7_y6_y5_y4_y3_y2_y1_y0, yF_yE_yD_yC_yB_yA_y9_y8);
__m256i yF_yE_yD_yC_yB_yA_y9_y8_y7_y6_y5_y4_y3_y2_y1_y0 = _mm256_permute4x64_epi64(
yF_yE_yD_yC_y7_y6_y5_y4_yB_yA_y9_y8_y3_y2_y1_y0,
_MM_SHUFFLE(3, 1, 2, 0));
_mm256_store_si256((__m256i *)&result[i], yF_yE_yD_yC_yB_yA_y9_y8_y7_y6_y5_y4_y3_y2_y1_y0);
}
}
int start = ((int)size / pixel_per_iteration) * pixel_per_iteration;
for (int i = start; i < size; i++)
{
result[i] =
0.2126 * img[(i * channels)]        
+ 0.7152 * img[(i * channels) + 1]  
+ 0.0722 * img[(i * channels) + 2]; 
}
}
