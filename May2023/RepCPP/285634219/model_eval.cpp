

#include <omp.h>
#include <math.h>
#include "support/common.h"

void call_RANSAC_kernel_block(int blocks, int threads, float *model_param_local,
flowvector *flowvectors, int flowvector_count, int max_iter, int error_threshold,
float convergence_threshold, int *g_out_id, int *model_candidate, int *outliers_candidate)
{
#pragma omp target teams num_teams(blocks) thread_limit(threads)
{
int outlier_block_count;
#pragma omp parallel 
{
const int tx         = omp_get_thread_num();
const int bx         = omp_get_team_num();
const int num_blocks = omp_get_num_teams();
const int block_dim  = omp_get_num_threads();

float vx_error, vy_error;
int   outlier_local_count = 0;

for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

const float *model_param = &model_param_local [4 * loop_count];

if(tx == 0) {
outlier_block_count = 0;
}
#pragma omp barrier

if(model_param[0] == -2011)
continue;

outlier_local_count = 0;

for(int i = tx; i < flowvector_count; i += block_dim) {
flowvector fvreg = flowvectors[i]; 
vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
(int)((fvreg.y - model_param[1]) * model_param[3])) - fvreg.vx;
vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
(int)((fvreg.x - model_param[0]) * model_param[3])) - fvreg.vy;
if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
outlier_local_count++;
}
}

#pragma omp atomic update
outlier_block_count += outlier_local_count;

#pragma omp barrier

if(tx == 0) {
if(outlier_block_count < flowvector_count * convergence_threshold) {
int index;
#pragma omp atomic capture
index = g_out_id[0]++;
model_candidate[index]    = loop_count;
outliers_candidate[index] = outlier_block_count;
}
}
}
}
}
}
