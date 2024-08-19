#include <iostream>
#include <omp.h>
#include <vector>
#include "AutoTest.h"
using namespace std;

long thread_result;
#pragma omp threadprivate(thread_result)

long run_semaphore(const vector<int>& first_array, const vector<int>& second_array) {
omp_lock_t lock;
omp_init_lock(&lock);
long result = 0;
#pragma omp parallel default(none) shared(first_array, second_array, result, lock)
#pragma omp for
for (int i = 0; i < first_array.size(); i++) {
long tmp = max(first_array[i] + second_array[i],  4 * first_array[i] - second_array[i]);
omp_set_lock(&lock);
if (tmp > 1)
result += tmp;
omp_unset_lock(&lock);
}
omp_destroy_lock(&lock);
return result;
}

long run_barrier(const vector<int>& first_array, const vector<int>& second_array) {
long result = 0;
#pragma omp parallel default(none) shared(first_array, second_array, result)
{
thread_result = 0;

#pragma omp for nowait
for (int i = 0; i < first_array.size() / 2; i++) {
long tmp = max(first_array[i] + second_array[i], 4 * first_array[i] - second_array[i]);
if (tmp > 1)
thread_result += tmp;
}

#pragma omp for nowait
for (int i = first_array.size() / 2; i < first_array.size(); i++) {
long tmp = max(first_array[i] + second_array[i], 4 * first_array[i] - second_array[i]);
if (tmp > 1)
thread_result += tmp;
}

#pragma omp critical
result += thread_result;
}
return result;
}

int main() {
auto tests = vector<AutoTest>();
auto s_time = omp_get_wtime();
#pragma omp parallel default(none) shared(tests)
#pragma omp for
for (int i = 2; i <= 7; i++) {
auto num_iter = static_cast<int>(pow(10, i));
auto value = AutoTest(num_iter);
#pragma omp critical
tests.push_back(value);
}
cout << "Spent on generation: " << omp_get_wtime() - s_time << " c\n\n";

double time;
for (const auto& test: tests) {
auto size = test.size;
int iterations = static_cast<int>(pow(10, 7) / size);
cout << "Size - " << size << endl;

time = test.run(iterations, run_semaphore);
cout << "\tTest semaphore: " << time << "c" << endl;

time = test.run(iterations, run_barrier);
cout << "\tTest barrier: " << time << "c" << endl;
}
cout << "Spent on start: " << omp_get_wtime() - s_time << " c\n";

return 0;
}
