

#include <iostream>
#include <chrono>

#define NUM_DIM 128
#define NUM_POINTS 10000

using namespace std;
using namespace std::chrono;

uint16_t distanceL1(uint8_t *x, uint8_t *y, uint16_t a, uint16_t b) {
uint16_t result = 0;

for (int k = 0; k < NUM_DIM; k++) {
result += abs(x[a * NUM_DIM + k] - y[b * NUM_DIM + k]);
}
return result;
}

int main() {
uint16_t i, j;

auto *m1 = (uint8_t *) aligned_alloc(256, NUM_POINTS * NUM_DIM * sizeof(uint8_t));
auto *m2 = (uint8_t *) aligned_alloc(256, NUM_POINTS * NUM_DIM * sizeof(uint8_t));
auto *m1m2Result = (uint16_t *) aligned_alloc(256,NUM_POINTS * sizeof(uint16_t));

for (i = 0; i < NUM_POINTS; i++) {
for (j = 0; j < NUM_DIM; j++) {
m1[i * NUM_DIM + j] = rand() % 256;
m2[i * NUM_DIM + j] = rand() % 256; 
}
}

auto start = std::chrono::high_resolution_clock::now();

uint16_t distance, mindistance;

#pragma omp parallel for default(shared)
for(i = 0; i < NUM_POINTS; i++) {
m1m2Result[i] = 0;
mindistance = distanceL1(m1, m2, i, 0);

for(j = 1; j < NUM_POINTS; j++) {
distance = distanceL1(m1, m2, i, j);

if (distance < mindistance) {
mindistance = distance;
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
