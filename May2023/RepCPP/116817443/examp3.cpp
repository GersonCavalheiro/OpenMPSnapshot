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
double *x, *y, *z;
std::default_random_engine generator(0);
std::uniform_real_distribution<double> distribution(0.0,1000.0);

x = static_cast<double *>(myalloc(64, SIZE * sizeof(double)));
for(unsigned int i = 0; i < SIZE; i++) {
x[i] = distribution(generator);
}

y = static_cast<double *>(myalloc(64, SIZE * sizeof(double)));
for(unsigned int i = 0; i < SIZE; i++) {
y[i] = distribution(generator);
}

z = static_cast<double *>(myalloc(64, SIZE * sizeof(double)));
for(unsigned int i = 0; i < SIZE; i++) {
z[i] = 0.0;
}

std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#pragma omp simd
for(unsigned int i = 0; i < SIZE; i++) {
z[i] = 2.0 * x[i] + y[i]; 
}
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
std::cout << "Loop calc took " << time_span.count() << " seconds." << std::endl;

double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for(unsigned int i = 0; i < SIZE; i++) {
sum += z[i];
}
std::cout << "Total sum over all elements is " << sum << std::endl;

myfree(x);
myfree(y);
myfree(z);
return 0;
}