

#include <math.h>
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>
#include "support/common.h"
#include "support/partitioner.h"

void run_cpu_threads(DATA_TYPE *matrix_out, DATA_TYPE *matrix, std::atomic_int *flags, int n, int m, int pad,
int n_threads, int ldim, int n_tasks, float alpha
#ifdef DYNAMIC_PARTITION
, std::atomic_int *worklist
#endif
) {

const int                REGS_CPU = REGS * ldim;
std::vector<std::thread> cpu_threads;
for(int i = 0; i < n_threads; i++) {

cpu_threads.push_back(std::thread([=]() {

#ifdef DYNAMIC_PARTITION
Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads, worklist);
#else
Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads);
#endif

const int matrix_size       = m * (n + pad);
const int matrix_size_align = (matrix_size + ldim * REGS - 1) / (ldim * REGS) * (ldim * REGS);

for(int my_s = cpu_first(&p); cpu_more(&p); my_s = cpu_next(&p)) {

DATA_TYPE   reg[REGS_CPU];
int pos      = matrix_size_align - 1 - (my_s * REGS_CPU);
int my_s_row = pos / (n + pad);
int my_x     = pos % (n + pad);
int pos2     = my_s_row * n + my_x;
#pragma unroll
for(int j = 0; j < REGS_CPU; j++) {
if(pos2 >= 0 && my_x < n && pos2 < matrix_size)
reg[j] = matrix[pos2];
else
reg[j] = 0;
pos--;
my_s_row = pos / (n + pad);
my_x     = pos % (n + pad);
pos2     = my_s_row * n + my_x;
}

while((&flags[my_s])->load() == 0) {
}
(&flags[my_s + 1])->fetch_add(1);

pos = matrix_size_align - 1 - (my_s * REGS_CPU);
#pragma unroll
for(int j = 0; j < REGS_CPU; j++) {
if(pos >= 0 && pos < matrix_size)
matrix_out[pos] = reg[j];
pos--;
}
}
}));
}
std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
