

#include <iostream>
#include <bitset>
#include <chrono>
#include <omp.h>

using namespace std::chrono;

#define R2(n) n, n + 2*64, n + 1*64, n + 3*64
#define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
#define REVERSE_BITS R6(0), R6(2), R6(1), R6(3)

uint64_t lookup[256] = {REVERSE_BITS};

uint64_t reverseBits(uint64_t n) {



uint64_t reverse = lookup[n & 0xff] << 56 |                
lookup[(n >> 8) & 0xff] << 48 |         
lookup[(n >> 16) & 0xff] << 40 |        
lookup[(n >> 24) & 0xff] << 32 |        
lookup[(n >> 32) & 0xff] << 24 |        
lookup[(n >> 40) & 0xff] << 16 |        
lookup[(n >> 48) & 0xff] << 8 |         
lookup[(n >> 56) & 0xff];               

return reverse;
}

int main() {
uint64_t *input_values;
input_values = (uint64_t *) aligned_alloc(sizeof(uint64_t), sizeof(uint64_t) * 100000000);
uint64_t *output_values;
output_values = (uint64_t *) aligned_alloc(sizeof(uint64_t), sizeof(uint64_t) * 100000000);

for (int i = 0; i < 100000000; i++) {
input_values[i] = i;
}

auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for default(shared)
for (int i = 0; i < 100000000; i++) {
output_values[i] = reverseBits(input_values[i]);
}

auto stop = std::chrono::high_resolution_clock::now();

std::bitset<64> x(input_values[26]);
std::cout << x << '\n';

std::bitset<64> y(output_values[26]);
std::cout << y << '\n';

std::cout << "Done in " << duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;
return 0;
}
