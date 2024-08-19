#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <string>
#include "utils.h"
typedef double (*integral_calc_func)(double x_start, double x_end, double dx);
double calc_integral(double x_start, double x_end, double dx) {
double ans = 0;
double x;
int i;
int n = ceil((x_end - x_start) / dx);
for (i = 0; i < n; ++i) {
x = i * dx + x_start;
ans += dx * 4.0 / (1.0 + x * x);
}
return ans;
}
double calc_integral_serial(double dx, integral_calc_func func) {
return func(0.0, 1.0, dx);
}
double calc_integral_par(double dx, integral_calc_func func) {
double global_ans = 0.0;
#pragma omp parallel
{
int num_threads = omp_get_num_threads();
int my_id = omp_get_thread_num();
double per_thread = 1.0 / (double)num_threads;
double my_start = my_id * per_thread;
double my_end = (1 + my_id) * per_thread;
double my_ans = func(my_start, my_end, dx);
#pragma omp atomic
global_ans += my_ans;
}
return global_ans;
}
double calc_integral_par_for(double dx) {
double ans = 0;
double x;
int i;
int n = ceil((1.0 - 0.0) / dx);
#pragma omp parallel for private(x) reduction(+ : ans)
for (i = 0; i < n; ++i) {
x = i * dx;
ans += dx * 4.0 / (1.0 + x * x);
}
return ans;
}
double calc_integral_avx2(double x_start, double x_end, double dx) {
__m256d _ans = _mm256_setzero_pd();
__m256d _four = _mm256_set1_pd(4.0);
__m256d _four_dx = _mm256_set1_pd(4.0 * dx);
__m256d _onetwothreefour_dx = _mm256_set_pd(0.0, 1.0 * dx, 2.0 * dx, 3.0 * dx);
__m256d _one = _mm256_set1_pd(1.0);
__m256d _dx = _mm256_set1_pd(dx);
__m256d _x_start = _mm256_set1_pd(x_start);
__m256d _x = _mm256_add_pd(
_mm256_set_pd(0.0, 1.0 * dx, 2.0 * dx, 3.0 * dx),  
_x_start);
while (_x[3] < x_end) {
_ans = _mm256_add_pd(_ans,
_mm256_div_pd(_four_dx,
_mm256_add_pd(_one, _mm256_mul_pd(_x, _x))));
_x = _mm256_add_pd(_x, _four_dx);
}
return _ans[0] + _ans[1] + _ans[2] + _ans[3];
}
double calc_integral_avx1(double x_start, double x_end, double dx) {
double ans = 0;
__m128d _ans = _mm_setzero_pd();
__m128d _two_dx = _mm_set1_pd(2.0 * dx);
__m128d _four_dx = _mm_set1_pd(4.0 * dx);
__m128d _one = _mm_set1_pd(1.0);
__m128d _dx = _mm_set1_pd(dx);
__m128d _x = _mm_add_pd(_mm_set_pd(0.0, 1.0 * dx), _mm_set1_pd(x_start));
while (_x[1] < x_end) {
_ans = _mm_add_pd(_ans, _mm_div_pd(_four_dx, _mm_add_pd(_one, _mm_mul_pd(_x, _x))));
_x = _mm_add_pd(_x, _two_dx);
}
return _ans[0] + _ans[1];
}
double calc_integral_omp_simd(double dx) {
double ans = 0;
double x;
int i;
int n = ceil(1.0 / dx);
#pragma omp simd reduction(+ : ans)
for (i = 0; i < n; ++i) {
x = i * dx;
ans += dx * 4.0 / (1.0 + x * x);
}
return ans;
}
double calc_integral_omp_simd_and_par_for(double dx) {
double ans = 0;
double x;
int i;
int n = ceil(1.0 / dx);
#pragma omp parallel for simd reduction(+ : ans)
for (i = 0; i < n; ++i) {
x = i * dx;
ans += dx * 4.0 / (1.0 + x * x);
}
return ans;
}
void do_comparison() {
double dx = 1e-8;
std::vector<single_func> vec_of_functions = {
[&dx]() { return calc_integral_serial(dx, calc_integral); },
[&dx]() { return calc_integral_par(dx, calc_integral); },
[&dx]() { return calc_integral_par_for(dx); },
[&dx]() { return calc_integral_serial(dx, calc_integral_avx1); },
[&dx]() { return calc_integral_par(dx, calc_integral_avx1); },
[&dx]() { return calc_integral_serial(dx, calc_integral_avx2); },
[&dx]() { return calc_integral_par(dx, calc_integral_avx2); },
[&dx]() { return calc_integral_omp_simd(dx); },
[&dx]() { return calc_integral_omp_simd_and_par_for(dx); },
};
std::vector<std::string> names = {
"Serial",
"Parallel",
"OMP Parallel For",
"AVX 128 Serial",
"AVX 128 Par",
"AVX 256 Serial",
"AVX 256 Par",
"OMP SIMD",
"OMP SIMD + FOR ",
};
double ans = calc_integral_serial(dx, calc_integral);
printf("Correct answer = %lf\n", ans);
Timer t(vec_of_functions, names, double_comparison, std::any(ans), 50);
t.run();
}
int main() {
do_comparison();
return 0;
}
