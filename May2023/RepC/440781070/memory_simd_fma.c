#include <immintrin.h>
#include <stdio.h>
void convert_memory_simd_fma(unsigned char *img, int width, int height, int channels, int threads, unsigned char *result)
{
int floats_per_operation = 4;
int size = width * height;
int pixel_per_thread_unaligned = size / threads;
int pixel_per_thread_aligned = ((int)pixel_per_thread_unaligned / floats_per_operation) * floats_per_operation;
__m128 r_factor = _mm_set_ps(0.2126, 0.2126, 0.2126, 0.2126);
__m128 g_factor = _mm_set_ps(0.7152, 0.7152, 0.7152, 0.7152);
__m128 b_factor = _mm_set_ps(0.0722, 0.0722, 0.0722, 0.0722);
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
__m128 r_vector, g_vector, b_vector, gray_vector;
__m128i gray_vector_int;
for (int i = pixel_per_thread_aligned * thread; i < end; i += floats_per_operation)
{
r_vector = _mm_set_ps(img[(i * channels)], img[(i + 1) * channels], img[(i + 2) * channels], img[(i + 3) * channels]);
g_vector = _mm_set_ps(img[(i * channels) + 1], img[(i + 1) * channels + 1], img[(i + 2) * channels + 1], img[(i + 3) * channels + 1]);
b_vector = _mm_set_ps(img[(i * channels) + 2], img[(i + 1) * channels + 2], img[(i + 2) * channels + 2], img[(i + 3) * channels + 2]);
gray_vector = _mm_setzero_ps();
gray_vector = _mm_fmadd_ps(r_vector, r_factor, gray_vector);
gray_vector = _mm_fmadd_ps(g_vector, g_factor, gray_vector);
gray_vector = _mm_fmadd_ps(b_vector, b_factor, gray_vector);
gray_vector_int = _mm_cvtps_epi32(gray_vector);
gray_vector_int = _mm_packus_epi32(gray_vector_int, gray_vector_int);
gray_vector_int = _mm_packus_epi16(gray_vector_int, gray_vector_int);
*(int *)(&result[i]) = _mm_cvtsi128_si32(gray_vector_int);
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