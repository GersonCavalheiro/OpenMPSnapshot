#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define BENCHMARK(name, block)\
do {\
clock_t t1, t2;\
t1 = clock();\
{\
block\
}\
t2 = clock();\
int duration_ms = (((double)t2 - t1) / CLOCKS_PER_SEC) * 1000;\
printf("%s - duration: %d ms\n", name, duration_ms);\
} while(0);
#define BENCHMARK_PARALL(name, block)\
do {\
double t1, t2;\
t1 = omp_get_wtime();\
{\
block\
}\
t2 = omp_get_wtime();\
double duration_ms = (t2 - t1) * 1000;\
printf("%s - duration: %lf ms\n", name, duration_ms);\
} while(0);
int is_ordered(const int *v, int n);
void print_array(int *v, int n);
void populate_array_seq(int *v, int n);
void populate_array_parall(int *v, int n);
void populate_array_parall_better(int *v, int n);
void mergesort_seq(int *v, int a, int b);
void mergesort_parall(int *v, int a, int b, int threads);
void merge(int *v, int a, int b);
#ifdef DEBUG
void test_is_ordered();
void test_merge();
void test_mergesort_seq();
void test_mergesort_parall();
void test_all();
#endif
void merge(int *v, int a, int b) {
int m = a + (b - a)/2;
int n = b - a;
int *tmp = (int*) malloc(n * sizeof(int));
int ti = 0, ai = a, bi = m;
while(ai < m && bi < b) {
tmp[ti++] = (v[ai] < v[bi]) ? v[ai++] : v[bi++];
}
while(ai < m) {
tmp[ti++] = v[ai++];
}
while(bi < b) {
tmp[ti++] = v[bi++];
}
memcpy(v + a, tmp, n * sizeof(int));
free(tmp);
}
void mergesort_seq(int *v, int a, int b) {
if(b - a <= 1)
return;
int m = a + (b - a)/2;
mergesort_seq(v, a, m);
mergesort_seq(v, m, b);
merge(v, a, b);
}
void mergesort_parall(int *v, int a, int b, int threads) {
if(b - a <= 1)
return;
int m = a + (b - a)/2;
if(threads > 1) {
#pragma omp parallel sections
{
#pragma omp section
mergesort_parall(v, a, m, threads / 2);
#pragma omp section
mergesort_parall(v, m, b, threads - threads / 2);
}
}
else {
mergesort_seq(v, a, m);
mergesort_seq(v, m, b);
}
merge(v, a, b);
}
int is_ordered(const int *v, int n) {
for(int i = 0; i < n - 1; ++i) {
if(v[i] > v[i+1])
return 0;
}
return 1;
}
void print_array(int *v, int n) {
for(int i = 0; i < n; ++i) {
printf("%d ", v[i]);
}
puts("");
}
void populate_array_seq(int *v, int n) {
for(int i = 0; i < n; ++i) {
v[i] = rand() % n;
}
}
void populate_array_parall(int *v, int n) {
#pragma omp parallel for
for(int i = 0; i < n; ++i) {
v[i] = rand() % n;
}
}
void populate_array_parall_better(int *v, int n) {
unsigned seed;
#pragma omp parallel private(seed)
{
seed = 25234 + 17 * omp_get_thread_num();
#pragma omp for
for(int i = 0; i < n; ++i) {
v[i] = rand_r(&seed) % n;
}
}
}
int main(int argc, char *argv[]) {
#ifdef DEBUG
test_all();
#endif
if(argc < 2) {
printf("Usage: %s <k>\n", argv[0]);
printf("Example: %s %d\n", argv[0], 28);
exit(0);
}
const int k = atoi(argv[1]);
const int n = 1 << k;
int *v = (int*) malloc(n * sizeof(int));
srand(time(NULL));
int num_threads;
#pragma omp parallel
{
#pragma omp single
num_threads = omp_get_num_threads();
}
printf("Running with %d threads\n", num_threads);
BENCHMARK_PARALL("populate_array_parall_better",
populate_array_parall_better(v, n);
);
BENCHMARK_PARALL("mergesort_parall",
mergesort_parall(v, 0, n, num_threads);
);
assert(is_ordered(v, n));
free(v);
return 0;
}
#ifdef DEBUG
void test_is_ordered() {
puts("test_is_ordered");
int v[] = {1, 2, 3, 4};
assert(is_ordered(v, 4));
int w[] = {1, 2, 4, 3};
assert(!is_ordered(w, 4));
}
void test_merge() {
puts("test_merge");
int v[] = {3, 4, 1, 2};
merge(v, 0, 4);
assert(is_ordered(v, 4));
int w[] = {1, 2, 3, 4};
merge(w, 0, 4);
assert(is_ordered(w, 4));
}
void test_mergesort_seq() {
puts("test_mergesort_seq");
int v[] = {5, 4, 3, 2, 1};
mergesort_seq(v, 0, 5);
assert(is_ordered(v, 5));
int w[] = {4, 3, 2, 1};
mergesort_seq(w, 0, 4);
assert(is_ordered(w, 4));
int x[] = {1, 2, 3};
mergesort_seq(x, 0, 3);
assert(is_ordered(x, 3));
}
void test_mergesort_parall() {
puts("test_mergesort_parall");
int v[] = {5, 4, 3, 2, 1};
mergesort_parall(v, 0, 5);
assert(is_ordered(v, 5));
int w[] = {4, 3, 2, 1};
mergesort_parall(w, 0, 4);
assert(is_ordered(w, 4));
int x[] = {1, 2, 3};
mergesort_parall(x, 0, 3);
assert(is_ordered(x, 3));
}
void test_all() {
puts("test");
test_is_ordered();
test_merge();
test_mergesort_seq();
test_mergesort_parall();
}
#endif
