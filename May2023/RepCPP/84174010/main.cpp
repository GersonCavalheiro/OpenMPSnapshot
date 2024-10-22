#include <iostream>
#include <omp.h>

void schedule_static();
void schedule_dynamic();
void schedule_auto();

int main() {

int i, N = 10;

#pragma omp parallel for num_threads(2) schedule(static)
for (i = 0; i < N; i++) {
if (omp_get_thread_num() == 0)
printf("Thread %d is doing iteration %d.\n", omp_get_thread_num( ), i);
}

#pragma omp parallel for num_threads(2) schedule(static)
for (i = 0; i < N; i++) {
if (omp_get_thread_num() == 1)
printf("Thread %d is doing iteration %d.\n", omp_get_thread_num( ), i);
}

#pragma omp parallel for num_threads(2) schedule(static, 4)
for (i = 0; i < N; i++) {
if (omp_get_thread_num() == 0)
printf("Thread %d is doing iteration %d.\n", omp_get_thread_num( ), i);
}

#pragma omp parallel for num_threads(2) schedule(dynamic, 3)
for (i = 0; i < N; i++) {
if (omp_get_thread_num() == 1)
printf("Thread %d is doing iteration %d.\n", omp_get_thread_num( ), i);
}

return 0;
}

void schedule_static() {
int N = 1000000000;
int i;
double avg = 0;

omp_set_num_threads(16);

double timer_started = omp_get_wtime();

#pragma omp parallel for reduction(+:avg) schedule(static)
for (i = 0; i < N; ++i) {
avg += i;
}

avg /= N;

double elapsed = omp_get_wtime() - timer_started;

std::cout << avg << " took " << elapsed << std::endl;
}

void schedule_dynamic() {
int N = 1000000000;
int i;
double avg = 0;

omp_set_num_threads(16);

double timer_started = omp_get_wtime();
#pragma omp parallel for reduction(+:avg) schedule(dynamic, 1000)
for (i = 0; i < N; ++i) {
avg += i;
}

avg /= N;

double elapsed = omp_get_wtime() - timer_started;

std::cout << avg << " took " << elapsed << std::endl;
}

void schedule_auto() {
int N = 1000000000;
int i;
double avg = 0;

omp_set_num_threads(16);

double timer_started = omp_get_wtime();

#pragma omp parallel for reduction(+:avg) schedule(auto)
for (i = 0; i < N; ++i) {
avg += i;
}

avg /= N;

double elapsed = omp_get_wtime() - timer_started;

std::cout << avg << " took " << elapsed << std::endl;
}
