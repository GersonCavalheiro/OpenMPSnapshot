#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <omp.h>
#include <math.h>
#include <x86intrin.h> 
#define DEBUGGING(_x)
void write_out(float ***a, int dim0, int dim1, int dim2) {
int i, j, k;
for (i = 0; i < dim0; i++) {
printf("Outer dimension number %d\n", i);
for (j = 0; j < dim1; j++) {
for (k = 0; k < dim2 - 1; k++) {
printf("%f, ", a[i][j][k]);
}
printf("%f\n", a[i][j][dim2 - 1]);
}
}
}
float ****new_empty_4d_matrix(int dim0, int dim1, int dim2, int dim3) {
float ****result = malloc(dim0 * sizeof(float ***));
float ***mat1 = malloc(dim0 * dim1 * sizeof(float **));
float **mat2 = malloc(dim0 * dim1 * dim2 * sizeof(float *));
float *mat3 = malloc(dim0 * dim1 * dim2 * dim3 * sizeof(float));
int i, j, k;
for (i = 0; i < dim0; i++) {
result[i] = &(mat1[i * dim1]);
for (j = 0; j < dim1; j++) {
result[i][j] = &(mat2[i * dim1 * dim2 + j * dim2]);
for (k = 0; k < dim2; k++) {
result[i][j][k] = &(mat3[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3]);
}
}
}
return result;
}
float ***new_empty_3d_matrix(int dim0, int dim1, int dim2) {
float ****mat4d;
float ***mat3d;
mat4d = new_empty_4d_matrix(1, dim0, dim1, dim2);
mat3d = mat4d[0];
free(mat4d);
return mat3d;
}
float ****copy_4d_matrix(float ****source_matrix, int dim0,
int dim1, int dim2, int dim3) {
int i, j, k, l;
float ****result = new_empty_4d_matrix(dim0, dim1, dim2, dim3);
for (i = 0; i < dim0; i++) {
for (j = 0; j < dim1; j++) {
for (k = 0; k < dim2; k++) {
for (l = 0; l < dim3; l++) {
result[i][j][k][l] = source_matrix[i][j][k][l];
}
}
}
}
return result;
}
float ****gen_random_4d_matrix(int dim0, int dim1, int dim2, int dim3) {
float ****result;
int i, j, k, l;
struct timeval seedtime;
int seed;
result = new_empty_4d_matrix(dim0, dim1, dim2, dim3);
gettimeofday(&seedtime, NULL);
seed = seedtime.tv_usec;
srandom(seed);
const int range = 1 << 16; 
const int bias = 1 << 12; 
float offset = 4.0;
for (i = 0; i < dim0; i++) {
for (j = 0; j < dim1; j++) {
for (k = 0; k < dim2; k++) {
for (l = 0; l < dim3; l++) {
long long rand = random();
int reduced_range = (rand % range);
float num = (((float) reduced_range) / ((float) bias)) + offset;
result[i][j][k][l] = num;
}
}
}
}
return result;
}
float ***gen_random_3d_matrix(int dim0, int dim1, int dim2) {
float ****mat4d;
float ***mat3d;
mat4d = gen_random_4d_matrix(1, dim0, dim1, dim2);
mat3d = mat4d[0];
free(mat4d);
return mat3d;
}
void check_result(float ***result, float ***control,
int dim0, int dim1, int dim2) {
int i, j, k;
double sum_abs_diff = 0.0;
const double EPSILON = 0.0625;
for (i = 0; i < dim0; i++) {
for (j = 0; j < dim1; j++) {
for (k = 0; k < dim2; k++) {
double diff = fabs(control[i][j][k] - result[i][j][k]);
assert(diff >= 0.0);
sum_abs_diff = sum_abs_diff + diff;
}
}
}
if (sum_abs_diff > EPSILON) {
fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
sum_abs_diff, EPSILON);
} else {
printf("COMMENT: sum of absolute differences (%f)  within acceptable range (%f)\n", sum_abs_diff, EPSILON);
}
}
void multichannel_conv(float ***image, float ****kernels, float ***output,
int width, int height, int nchannels, int nkernels,
int kernel_order) {
int h, w, x, y, c, m;
for (m = 0; m < nkernels; m++) {
for (w = 0; w < width; w++) {
for (h = 0; h < height; h++) {
float sum = 0.0;
for (c = 0; c < nchannels; c++) {
for (x = 0; x < kernel_order; x++) {
for (y = 0; y < kernel_order; y++) {
sum += image[w + x][h + y][c] * kernels[m][c][x][y];
}
}
output[m][w][h] = sum;
}
}
}
}
}
void team_conv(float ***image, float ****kernels, float ***output,
int width, int height, int nchannels, int nkernels,
int kernel_order) {
int m_kernels = nkernels, m_width = width, m_height = height, m_c = nchannels, m_ko = kernel_order;
int m;
#pragma omp parallel for
for (m = 0; m < m_kernels; m++)
{
int w;
for (w = 0; w < m_width; w++)
{
int h;
for (h = 0; h < m_height; h++)
{
float sum = 0.0;
int c;
for (c = 0; c < m_c; c++)
{
int x, y;
for (x = 0; x < m_ko; x++)
{
for (y = 0; y < m_ko - 1; y += 2)
{
sum += image[w + x][h + y][c] * kernels[m][c][x][y];
sum += image[w + x][h + y + 1][c] * kernels[m][c][x][y + 1];
}
sum += image[w + x][h + y][c] * kernels[m][c][x][y];
}
}
output[m][w][h] = sum;
}
}
}
}
int main(int argc, char **argv) {
float ***image, ****kernels, ***output;
float ***control_output;
long long mul_time;
int width, height, kernel_order, nchannels, nkernels;
struct timeval start_time;
struct timeval stop_time;
if (argc != 6) {
fprintf(stderr,
"Usage: conv-harness <image_width> <image_height> <kernel_order> <number of channels> <number of kernels>\n");
exit(1);
} else {
width = atoi(argv[1]);
height = atoi(argv[2]);
kernel_order = atoi(argv[3]);
nchannels = atoi(argv[4]);
nkernels = atoi(argv[5]);
}
switch (kernel_order) {
case 1:
case 3:
case 5:
case 7:
break;
default:
fprintf(stderr, "FATAL: kernel_order must be 1, 3, 5 or 7, not %d\n",
kernel_order);
exit(1);
}
image = gen_random_3d_matrix(width + kernel_order, height + kernel_order,
nchannels);
kernels = gen_random_4d_matrix(nkernels, nchannels, kernel_order, kernel_order);
output = new_empty_3d_matrix(nkernels, width, height);
control_output = new_empty_3d_matrix(nkernels, width, height);
multichannel_conv(image, kernels, control_output, width,
height, nchannels, nkernels, kernel_order);
gettimeofday(&start_time, NULL);
team_conv(image, kernels, output, width,
height, nchannels, nkernels, kernel_order);
gettimeofday(&stop_time, NULL);
mul_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
(stop_time.tv_usec - start_time.tv_usec);
printf("Team conv time: %lld microseconds\n", mul_time);
DEBUGGING(write_out(output, nkernels, width, height));
check_result(output, control_output, nkernels, width, height);
return 0;
}
