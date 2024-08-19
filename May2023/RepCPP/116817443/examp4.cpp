#include <iostream>
#include <random>
#include <chrono>

#define SIZE (64l * 39062500l) 

#if defined __INTEL_COMPILER
#define myalloc(x,y) _mm_malloc(y,x)
#define myfree(x) _mm_free(x)
#elif defined __GNUC__
#define myalloc(x,y) aligned_alloc(x,y)
#define myfree(x) free(x)
#endif

int main() {
double *x;
std::default_random_engine generator(0);
std::uniform_real_distribution<double> distribution(0.0,1000.0);

x = static_cast<double *>(myalloc(64, SIZE * sizeof(double)));
for(unsigned int i = 0; i < SIZE; i++) {
x[i] = distribution(generator);
}
double sum = 0.0;

std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp simd reduction(+:sum)
for(unsigned int i = 0; i < SIZE; i++) {
sum += 2.0 * x[i];
}
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
std::cout << "Loop calc took " << time_span.count() << " seconds." << std::endl;

std::cout << "Total sum over all elements is " << sum << std::endl;

myfree(x);
return 0;
}