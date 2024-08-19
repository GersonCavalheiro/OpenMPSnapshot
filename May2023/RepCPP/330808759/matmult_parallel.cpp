
#ifdef _OPENMP

#include <omp.h>

#endif

#include <cstdio>
#include <math.h>


using namespace std;


void mult_basic(long *result, long *a, long *b, int height, int width, int widthA) {

#pragma omp parallel for
for (int row = 0; row < height; row++) {
for (int i = 0; i < widthA; i++) {
for (int col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_basic_threads_size(long *result, long *a, long *b, int height, int width, int widthA, int threads) {

omp_set_num_threads(threads);
#pragma omp parallel for
for (int row = 0; row < height; row++) {
for (int i = 0; i < widthA; i++) {
for (int col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_3for(long *result, long *a, long *b, int height, int width, int widthA) {

omp_set_nested(1);
#pragma omp parallel for
for (int row = 0; row < height; row++) {
#pragma omp parallel for
for (int i = 0; i < widthA; i++) {
#pragma omp parallel for
for (int col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_basic_private(long *result, long *a, long *b, int height, int width, int widthA) {
int row, col, i;
#pragma omp parallel for private(row, i, col)
for (row = 0; row < height; row++) {
for (i = 0; i < widthA; i++) {
for (col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_basic_private_shared(long *result, long *a, long *b, int height, int width, int widthA) {
int row, col, i;
#pragma omp parallel for private(row, i, col) shared(a, b, result)
for (row = 0; row < height; row++) {
for (i = 0; i < widthA; i++) {
for (col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_basic_private_private(long *result, long *a, long *b, int height, int width, int widthA) {
int row, col, i;
#pragma omp parallel for private(row, i, col) firstprivate (a, b) shared (result)
for (row = 0; row < height; row++) {
for (i = 0; i < widthA; i++) {
for (col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_basic_private_switchedloops(long *result, long *a, long *b, int height, int width, int widthA) {
int row, col, i;

#pragma omp parallel for private(row, i, col)
for (row = 0; row < height; row++) {
for (col = 0; col < width; col++) {
for (i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_collapse3(long *result, long *a, long *b, int height, int width, int widthA) {

#pragma omp parallel for collapse(3)
for (int row = 0; row < height; row++) {
for (int i = 0; i < widthA; i++) {
for (int col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_collapse2(long *result, long *a, long *b, int height, int width, int widthA) {

#pragma omp parallel for collapse(2)
for (int row = 0; row < height; row++) {
for (int i = 0; i < widthA; i++) {
for (int col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_collapse2_private(long *result, long *a, long *b, int height, int width, int widthA) {

int row, col, i;
#pragma omp parallel for collapse(2) private(row, col, i)
for (row = 0; row < height; row++) {
for (i = 0; i < widthA; i++) {
for (col = 0; col < width; col++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_collapse2_private_cache(long *result, long *a, long *b, int height, int width, int widthA) {
int row, col, i;
#pragma omp parallel for collapse(2) private(row, col, i)
for (row = 0; row < height; row++) {
for (col = 0; col < width; col++) {
long sum = 0;
for (i = 0; i < widthA; i++) {
sum += a[row * widthA + i] * b[i * width + col];
}
result[row * width + col] = sum;
}
}
}
void mult_collapse2_private_cache_threads_size(long *result, long *a, long *b, int height, int width, int widthA, int threads) {
omp_set_num_threads(threads);
int row, col, i;
#pragma omp parallel for collapse(2) private(row, col, i)
for (row = 0; row < height; row++) {
for (col = 0; col < width; col++) {
long sum = 0;
for (i = 0; i < widthA; i++) {
sum += a[row * widthA + i] * b[i * width + col];
}
result[row * width + col] = sum;
}
}
}


void mult_reduction(long *result, long *a, long *b, int height, int width, int widthA) {
long sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
sum = 0;
for (int i = 0; i < widthA; i++) {
sum += a[row * widthA + i] * b[i * width + col];
}
result[row * width + col] = sum;
}
}
}


void mult_reduction_threads_size(long *result, long *a, long *b, int height, int width, int widthA, int threads) {
omp_set_num_threads(threads);
long sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
sum = 0;
for (int i = 0; i < widthA; i++) {
sum += a[row * widthA + i] * b[i * width + col];
}
result[row * width + col] = sum;
}
}
}

void mult_reduction_inner(long *result, long *a, long *b, int height, int width, int widthA) {
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
long sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < widthA; i++) {
sum += a[row * widthA + i] * b[i * width + col];
}
result[row * width + col] = sum;
}
}
}

void mult_reduction_inner_collapse(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for collapse(2)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
long sum = 0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < widthA; i++) {
sum += a[row * widthA + i] * b[i * width + col];
}
result[row * width + col] = sum;
}
}
}


void mult_schedule_static(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(static)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_static_chunk(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(static, 125)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_static_chunk_dynamic(long *result, long *a, long *b, int height, int width, int widthA) {
int chunk = ceil(height/omp_get_num_procs());
#pragma omp parallel for schedule(static, chunk)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_dynamic(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(dynamic)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_dynamic_chunk(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(dynamic, 4)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_guided(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(guided)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_guided_chunk(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(guided, 125)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}


void mult_schedule_auto(long *result, long *a, long *b, int height, int width, int widthA) {
#pragma omp parallel for schedule(auto)
for (int row = 0; row < height; row++) {
for (int col = 0; col < width; col++) {
for (int i = 0; i < widthA; i++) {
result[row * width + col] += a[row * widthA + i] * b[i * width + col];
}
}
}
}




