

#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>
#include "kernel.h"
#include "support/partitioner.h"

void run_cpu_threads(T *output, T *input, std::atomic_int *flags, int size, int value, int n_threads, int ldim,
int n_tasks, float alpha
#ifdef DYNAMIC_PARTITION
, std::atomic_int *worklist
#endif
) {

const int REGS_CPU = REGS * ldim;
std::vector<std::thread> cpu_threads;
for(int i = 0; i < n_threads; i++) {
cpu_threads.push_back(std::thread([=]() {

#ifdef DYNAMIC_PARTITION
Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads, worklist);
#else
Partitioner p = partitioner_create(n_tasks, alpha, i, n_threads);
#endif

for(int my_s = cpu_first(&p); cpu_more(&p); my_s = cpu_next(&p)) {

int l_count = 0;
T   reg[REGS_CPU];
int pos = my_s * REGS_CPU;
#pragma unroll
for(int j = 0; j < REGS_CPU; j++) {
if(pos < size) {
reg[j] = input[pos];
if(reg[j] != value)
l_count++;
} else
reg[j] = value;
pos++;
}

int p_count;
while((p_count = (&flags[my_s])->load()) == 0) {
}
(&flags[my_s + 1])->fetch_add(p_count + l_count);
l_count = p_count - 1;

pos = l_count;
#pragma unroll
for(int j = 0; j < REGS_CPU; j++) {
if(reg[j] != value) {
output[pos] = reg[j];
pos++;
}
}
}
}));
}
std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
