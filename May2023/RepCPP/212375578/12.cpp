#include <cstdio>
#include <unistd.h>
#include <omp.h>

const int threads = 8;

void way1() {
for (int i = threads - 1; i >= 0; i--) {
#pragma omp parallel
{
if (omp_get_thread_num() == i) {
printf("T%d/%d: Hello World\n",
omp_get_thread_num(),
omp_get_num_threads());
}
}
}
}

void way2() {
#pragma omp parallel
{
usleep(200 * (omp_get_num_threads() - omp_get_thread_num()));
printf("T%d/%d: Hello World\n",
omp_get_thread_num(),
omp_get_num_threads());
}
}

void way3() {
omp_lock_t* locks = new omp_lock_t[threads];
for (int i = 0; i < threads; i++) {
omp_init_lock(&locks[i]);
}

for (int i = threads - 2; i >= 0; i--) {
omp_set_lock(&locks[i]);
}

#pragma omp parallel
{
omp_set_lock(&locks[omp_get_thread_num()]);
printf("T%d/%d: Hello World\n",
omp_get_thread_num(),
omp_get_num_threads());
omp_unset_lock(&locks[omp_get_thread_num()]);
omp_unset_lock(&locks[omp_get_thread_num() - 1]);
}
}

int main() {
omp_set_num_threads(threads);
way1();
}
