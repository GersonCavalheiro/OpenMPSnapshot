#include <iostream>
#include <thread>
#include <vector>
#include <omp.h>

uint64_t A = 134775813;
uint64_t B = 1;
uint64_t C = 2147483648; 

size_t N = 50000000;
uint64_t seed = 0;

uint64_t x_min = 0;
uint64_t x_max = 100;

#define CACHE_LINE 64u

typedef struct element_t_
{
alignas(CACHE_LINE) unsigned value;
} element_t;

void set_num_threads(unsigned T);
unsigned get_num_threads();

unsigned randomize(uint64_t* V, size_t N, uint64_t seed, uint64_t min, uint64_t max) {
unsigned T = get_num_threads();
std::vector<uint64_t> mult;
mult.reserve(T);
mult.emplace_back(A);
for (unsigned i = 1; i < T + 1; i++)
mult.emplace_back(mult.back() * A % C);

uint64_t sum = 0;
std::vector<element_t> partial_sum(T, element_t{ 0 });
std::vector<std::thread> threads;

auto thread_proc = [T, V, N, seed, &partial_sum, &mult, min, max](unsigned t) {
uint64_t At = mult.back();
uint64_t D = (B * (At - 1) / (A - 1)) % C;
uint64_t x = (seed * mult[t] + B * (mult[t] - 1) / (A - 1)) % C;
uint64_t accumulated = 0;
for (unsigned i = t; i < N; i += T) {
V[i] = x % (max - min) + min;
accumulated += V[i];
x = (x * At + D) % C;
}
partial_sum[t].value = accumulated;
};

for (unsigned t = 0; t < T; ++t)
threads.emplace_back(thread_proc, t);
thread_proc(0);
for (auto& thread : threads)
thread.join();

for (unsigned i = 0; i < T; ++i)
sum += partial_sum[i].value;
return sum / N;
}

typedef struct experiment_result_ {
unsigned result;
double time;
} experiment_result;

void run_experiments(uint64_t* V, experiment_result* results, size_t N, uint64_t seed) {
unsigned T = (unsigned)std::thread::hardware_concurrency();

for (unsigned i = 0; i < T; ++i) {
double t0 = omp_get_wtime();
set_num_threads(i + 1);
results[i].result = randomize(V, N, seed, x_min, x_max);
results[i].time = omp_get_wtime() - t0;
}
}

int main() {
unsigned T = std::thread::hardware_concurrency();
uint64_t* V = new uint64_t[N];
experiment_result* results = (experiment_result*)malloc(T * sizeof(experiment_result));
run_experiments(V, results, N, seed);
#pragma warning(disable : 4996)
freopen("generator_output.txt", "w", stdout);

printf("Thread\tResult\tTime\n");
for (unsigned t = 0; t < T; t++)
printf("%u\t%u\t%f\n", t + 1, results[t].result, results[t].time);
free(results);
fclose(stdout);
return 0;
}