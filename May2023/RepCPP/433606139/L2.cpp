

#include <iostream>
#include <ctgmath>
#include <chrono>
#include <immintrin.h>

#define NUM_DIM 128
#define NUM_POINTS 10000

using namespace std;
using namespace std::chrono;

float distanceL2(float *x, float *y, float *values, uint16_t a, uint16_t b) {
int k, i;
__m256 t0, t1, diff, pwr, sum;

sum = _mm256_setzero_ps();

for (k = 0; k < NUM_DIM; k = k + 8) {
t0 = _mm256_load_ps(&x[a * NUM_DIM + k]);
t1 = _mm256_load_ps(&y[b * NUM_DIM + k]);
diff = _mm256_sub_ps(t0, t1);
pwr = _mm256_mul_ps(diff, diff);
sum = _mm256_add_ps(sum, pwr);
}

_mm256_store_ps(values, sum);

return sqrt(values[0] + values[1] + values[2] + values[3] + values[4] + values[5] + values[6] + values[7]);
}

int main() {
uint16_t i, j;

float *m1 = (float *) aligned_alloc(32, NUM_POINTS * NUM_DIM * sizeof(float));
float *m2 = (float *) aligned_alloc(32, NUM_POINTS * NUM_DIM * sizeof(float));
uint16_t *m1m2Result = (uint16_t *) aligned_alloc(16, NUM_POINTS * sizeof(uint16_t));
float *values = (float *) aligned_alloc(32, 8 * sizeof(float));

for (i = 0; i < NUM_POINTS; i++) {
for (j = 0; j < NUM_DIM; j++) {
m1[i * NUM_DIM + j] = rand();
m2[i * NUM_DIM + j] = rand();
}
}

auto start = std::chrono::high_resolution_clock::now();

float distance, minDistance;

#pragma omp parallel for default(shared)
for (i = 0; i < NUM_POINTS; i++) {

m1m2Result[i] = 0;
minDistance = distanceL2(m1, m2, values, i, 0);

for (j = 1; j < NUM_POINTS; j++) {

distance = distanceL2(m1, m2, values, i, j);

if (distance < minDistance) {
minDistance = distance;
m1m2Result[i] = j;
}
}
}

auto stop = std::chrono::high_resolution_clock::now();
std::cout << "Done in " << duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;



free(m1);
free(m2);
free(m1m2Result);
return 0;
}
