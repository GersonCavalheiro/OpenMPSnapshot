


#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include "gaussian.h"

#pragma omp declare target 
void invert(float* data, int actualsize, float* log_determinant)  {
int maxsize = actualsize;
int n = actualsize;

*log_determinant = 0.0;

if (actualsize == 1) {
*log_determinant = logf(data[0]);
data[0] = 1.f / data[0];
} else {

for (int i=1; i < actualsize; i++) data[i] /= data[0]; 
for (int i=1; i < actualsize; i++)  { 
for (int j=i; j < actualsize; j++)  { 
float sum = 0.0;
for (int k = 0; k < i; k++)  
sum += data[j*maxsize+k] * data[k*maxsize+i];
data[j*maxsize+i] -= sum;
}
if (i == actualsize-1) continue;
for (int j=i+1; j < actualsize; j++)  {  
float sum = 0.0;
for (int k = 0; k < i; k++)
sum += data[i*maxsize+k]*data[k*maxsize+j];
data[i*maxsize+j] = 
(data[i*maxsize+j]-sum) / data[i*maxsize+i];
}
}

for(int i=0; i<actualsize; i++) {
*log_determinant += logf(fabs(data[i*n+i]));
}

for ( int i = 0; i < actualsize; i++ )  
for ( int j = i; j < actualsize; j++ )  {
float x = 1.f;
if ( i != j ) {
x = 0.0;
for ( int k = i; k < j; k++ ) 
x -= data[j*maxsize+k]*data[k*maxsize+i];
}
data[j*maxsize+i] = x / data[j*maxsize+j];
}
for ( int i = 0; i < actualsize; i++ )   
for ( int j = i; j < actualsize; j++ )  {
if ( i == j ) continue;
float sum = 0.0;
for ( int k = i; k < j; k++ )
sum += data[k*maxsize+j]*( (i==k) ? 1.f : data[i*maxsize+k] );
data[i*maxsize+j] = -sum;
}
for ( int i = 0; i < actualsize; i++ )   
for ( int j = 0; j < actualsize; j++ )  {
float sum = 0.0;
for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
sum += ((j==k)?1.f:data[j*maxsize+k])*data[k*maxsize+i];
data[j*maxsize+i] = sum;
}
}
}

#pragma omp end declare target 

#pragma omp declare target 
void compute_row_col(const int bid, const int n, int* row, int* col) {
int i = 0;
for(int r=0; r < n; r++) {
for(int c=0; c <= r; c++) {
if(i == bid) {  
*row = r;
*col = c;
return;
}
i++;
}
}
}
#pragma omp end declare target 


#pragma omp declare target 
void constants_kernel(
const float *clusters_R,
float *clusters_Rinv,
const float *clusters_N,
float *clusters_pi,
float *clusters_constant,
float *clusters_avgvar,
float *matrix,
float *determinant_arg,
float *sum,
const int num_clusters, 
const int num_dimensions) {


int tid = omp_get_thread_num(); 
int bid = omp_get_team_num(); 
int num_threads = omp_get_num_threads();
int num_elements = num_dimensions*num_dimensions;

float log_determinant;



for(int i=tid; i<num_elements; i+= num_threads ) {
matrix[i] = clusters_R[bid*num_dimensions*num_dimensions+i];
}

#pragma omp barrier 

if(tid == 0) { 
#if DIAG_ONLY
determinant_arg[0] = 1.0f;
for(int i=0; i < num_dimensions; i++) {
determinant_arg[0] *= matrix[i*num_dimensions+i];
matrix[i*num_dimensions+i] = 1.0f / matrix[i*num_dimensions+i];
}
determinant_arg[0] = logf(determinant_arg[0]);
#else 
invert(matrix,num_dimensions,determinant_arg);
#endif
}
#pragma omp barrier 
log_determinant = determinant_arg[0];

for(int i=tid; i<num_elements; i+= num_threads) {
clusters_Rinv[bid*num_dimensions*num_dimensions+i] = matrix[i];
}

#pragma omp barrier

if(tid == 0) {
clusters_constant[bid] = -num_dimensions*0.5f*logf(2.0f*PI) - 0.5f*log_determinant;
}

#pragma omp barrier

if(bid == 0) {

if(tid == 0) {
sum[0] = 0.0;
for(int i=0; i<num_clusters; i++) {
sum[0] += clusters_N[i];
}
}

#pragma omp barrier

for(int i = tid; i < num_clusters; i += num_threads) {
if(clusters_N[i] < 0.5f) {
clusters_pi[tid] = 1e-10;
} else {
clusters_pi[tid] = clusters_N[i] / sum[0];
}
}
}
}
#pragma omp end declare target 

#pragma omp declare target 
void estep1_kernel(
float *data, 
const float* clusters_Rinv, 
float *clusters_memberships, 
float *clusters_pi, 
float *clusters_constant,
float *clusters_means, 
float *means,
float *Rinv,
const int num_dimensions, 
const int num_events) 
{
int c = omp_get_team_num() / NUM_BLOCKS;


int num_pixels_per_block = num_events / NUM_BLOCKS;
num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);

const unsigned int tid = omp_get_thread_num();
const unsigned int groupIdx_y = omp_get_team_num() % NUM_BLOCKS;
const unsigned int groupDim_y = NUM_BLOCKS;

int start_index = groupIdx_y * num_pixels_per_block + tid;

int end_index;
if(groupIdx_y == groupDim_y-1) {
end_index = num_events;
} else { 
end_index = (groupIdx_y+1) * num_pixels_per_block;
}


if(tid < num_dimensions) {
means[tid] = clusters_means[c*num_dimensions+tid];
}

for(int i=tid; i < num_dimensions*num_dimensions; i+= NUM_THREADS_ESTEP) {
Rinv[i] = clusters_Rinv[c*num_dimensions*num_dimensions+i]; 
}

float cluster_pi = clusters_pi[c];
float constant = clusters_constant[c];

#pragma omp barrier

for(int event=start_index; event<end_index; event += NUM_THREADS_ESTEP) {
float like = 0.0f;
#if DIAG_ONLY
for(int j=0; j<num_dimensions; j++) {
like += (data[j*num_events+event]-means[j]) * 
(data[j*num_events+event]-means[j]) * Rinv[j*num_dimensions+j];
}
#else
for(int i=0; i<num_dimensions; i++) {
for(int j=0; j<num_dimensions; j++) {
like += (data[i*num_events+event]-means[i]) * 
(data[j*num_events+event]-means[j]) * Rinv[i*num_dimensions+j];
}
}
#endif
clusters_memberships[c*num_events+event] = -0.5f * like + constant + logf(cluster_pi);
}
}
#pragma omp end declare target 

#pragma omp declare target 
void estep2_kernel(
float* clusters_memberships,
float* likelihood,
float* total_likelihoods, 
const int num_dimensions, 
const int num_clusters, 
const int num_events ) 
{
float temp;
float thread_likelihood = 0.0f;
float max_likelihood;
float denominator_sum;

int groups = omp_get_num_teams(); 
int tid = omp_get_thread_num();
int bid = omp_get_team_num();

int num_pixels_per_block = num_events / groups; 
num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);

int start_index = bid * num_pixels_per_block + tid;

int end_index;
if(bid == groups-1) {
end_index = num_events;
} else {
end_index = (bid+1) * num_pixels_per_block;
}

total_likelihoods[tid] = 0.0;



for(int pixel=start_index; pixel<end_index; pixel += NUM_THREADS_ESTEP) {
max_likelihood = clusters_memberships[pixel];
for(int c=1; c<num_clusters; c++) {
max_likelihood = fmaxf(max_likelihood,clusters_memberships[c*num_events+pixel]);
}

denominator_sum = 0.0;
for(int c=0; c<num_clusters; c++) {
temp = expf(clusters_memberships[c*num_events+pixel]-max_likelihood);
denominator_sum += temp;
}
denominator_sum = max_likelihood + logf(denominator_sum);
thread_likelihood += denominator_sum;

for(int c=0; c<num_clusters; c++) {
clusters_memberships[c*num_events+pixel] = 
expf(clusters_memberships[c*num_events+pixel] - denominator_sum);
}
}

total_likelihoods[tid] = thread_likelihood;
#pragma omp barrier

for (unsigned int bit = NUM_THREADS_ESTEP >> 1; bit > 0; bit >>= 1) {
float t = total_likelihoods[tid] + total_likelihoods[tid^bit];
#pragma omp barrier
total_likelihoods[tid] = t;
#pragma omp barrier
}

if(tid == 0) {
likelihood[bid] = total_likelihoods[tid];
}
}
#pragma omp end declare target 



#endif 
