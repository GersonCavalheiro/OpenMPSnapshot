

#include "kernel.h"

void call_TaskQueue_gpu(int blocks,
int threads,
const task_t *__restrict task_queue,
int *__restrict data_queue,
int *__restrict consumed, 
int iterations,
int offset,
int gpuQueueSize)
{
#pragma omp target teams num_teams(blocks) thread_limit(threads)
{
int next[3];
#pragma omp parallel 
{
task_t* t = (task_t*)&next[1];

const int tid       = omp_get_thread_num();
const int tile_size = omp_get_num_threads();

if(tid == 0) {
#pragma omp atomic capture
*next = (*consumed)++;
t->id = task_queue[*next].id;
t->op = task_queue[*next].op;
}

#pragma omp barrier

while(*next < gpuQueueSize) {
if(t->op == SIGNAL_WORK_KERNEL) {
for(int i = 0; i < iterations; i++) {
data_queue[(t->id - offset) * tile_size + tid] += tile_size;
}

data_queue[(t->id - offset) * tile_size + tid] += t->id;
}
if(t->op == SIGNAL_NOTWORK_KERNEL) {
for(int i = 0; i < 1; i++) {
data_queue[(t->id - offset) * tile_size + tid] += tile_size;
}

data_queue[(t->id - offset) * tile_size + tid] += t->id;
}
if(tid == 0) {
#pragma omp atomic capture
*next = (*consumed)++;
t->id = task_queue[*next].id;
t->op = task_queue[*next].op;
}
#pragma omp barrier
}
}
}
}
