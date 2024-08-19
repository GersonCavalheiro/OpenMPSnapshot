#include <omp.h>
#include <chrono>
#include <string>
#include <vector>
#include <stdio.h>
#include <cmath>
#include <numeric>
std::vector<int> get_vector(int nums){
std::vector<int> v;
for (int i=0; i< nums; ++i){
v.push_back(i);
}
return v;
}
int do_sum_manual(const std::vector<int> &v);
int do_sum_serial(const std::vector<int> &v);
int do_sum_par_for(const std::vector<int> &v);
int faster_for(const std::vector<int> &v);
int for_reduction(const std::vector<int> &v);
typedef int (*function) (const std::vector<int>&);
int main(){
int N = 10000000;
int experiments = 100;
std::vector<function> vec_of_functions = {
do_sum_serial,
do_sum_manual,
do_sum_par_for, 
faster_for,
for_reduction
};
std::vector<std::string> names = {
"Serial",
"Manual",
"Parallel For",
"Faster For",
"For Reduction"
};
std::vector<double> times(vec_of_functions.size(), 0.0);
std::vector<int> values(vec_of_functions.size(), 0);
auto vector_to_use = get_vector(N);
for (int f =0; f < vec_of_functions.size(); ++f){
for (int i=0; i < experiments; ++i){
auto start = std::chrono::high_resolution_clock::now();
values[f] += vec_of_functions[f](vector_to_use);
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration<double, std::milli>(end - start).count();
times[f] += duration;
}
}
int correct_answer = std::accumulate(vector_to_use.cbegin(), vector_to_use.cend(), 0);
for (int f = 0; f < vec_of_functions.size(); ++f){
bool is_correct = correct_answer * experiments == values[f];
printf("Method %-20s. Correct Answer = %-10s. Time = %lfms\n", names[f].c_str(), is_correct ? "YES" : "NO", times[f] / experiments);
}
}
int do_sum_serial(const std::vector<int> &v){
int total = 0;
for (auto &i: v){
total += i;
}
return total;
}
int do_sum_manual(const std::vector<int> &v){
int N = v.size();
std::vector<int> totals (omp_get_max_threads(), 0);
#pragma omp parallel
{
int num_threads = omp_get_num_threads();
int per_thread = ceil((float)N / num_threads);
int my_thread_num = omp_get_thread_num();
int my_start = per_thread * my_thread_num;
int my_end = my_start + per_thread;
for (int i=my_start; i < my_end; ++i){
totals[my_thread_num] += v[i];
}
}
int total = 0;
for (auto i: totals) total += i;
return total;
}
int do_sum_par_for(const std::vector<int> &v){
int N = v.size();
int global_total = 0;
std::vector<int> totals (omp_get_max_threads(), 0);
#pragma omp parallel
{
int thread_num = omp_get_thread_num();
#pragma omp for
for (int i=0; i < N; ++i){
totals[thread_num] += v[i];
}
}
int total = 0;
for (auto i: totals) total += i;
return total;
}
int faster_for(const std::vector<int> &v){
int N = v.size();
int global_total = 0;
#pragma omp parallel
{
int my_total = 0;
#pragma omp for
for (int i=0; i < N; ++i){
my_total += v[i];
}
#pragma omp atomic
global_total += my_total;
}
return global_total;
}
int for_reduction(const std::vector<int> &v){
int N = v.size();
int global_total = 0;
#pragma omp parallel for reduction(+:global_total)
for (int i=0; i < N; ++i){
global_total += v[i];
}
return global_total;
}
