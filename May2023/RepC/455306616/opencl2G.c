#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <CL/cl.h>
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#define N_THREADS (2)
#define WORKGROUP_SIZE 128
#define MAX_SOURCE_SIZE 16384
typedef unsigned char byte_t;
int seed;
byte_t *img_load(char *img_file, int *width, int *height, int *n_channels)
{
byte_t *data;
data = stbi_load(img_file, width, height, n_channels, 0);
if (data == NULL) {
fprintf(stderr, "ERROR LOADING IMAGE: << Invalid file name or format >> \n");
exit(EXIT_FAILURE);
}
return data;
}
void img_save(char *img_file, byte_t *data, int width, int height, int n_channels)
{
char *ext;
ext = strrchr(img_file, '.');
if (!ext) {
fprintf(stderr, "ERROR SAVING IMAGE: << Unspecified format >> \n\n");
return;
}
printf("Savin Image...\n");
if ((strcmp(ext, ".jpeg") == 0) || (strcmp(ext, ".jpg") == 0)) {
stbi_write_jpg(img_file, width, height, n_channels, data, 100);
} else if (strcmp(ext, ".png") == 0) {
stbi_write_png(img_file, width, height, n_channels, data, width * n_channels);
} else if (strcmp(ext, ".bmp") == 0) {
stbi_write_bmp(img_file, width, height, n_channels, data);
} else if (strcmp(ext, ".tga") == 0) {
stbi_write_tga(img_file, width, height, n_channels, data);
} else {
fprintf(stderr, "ERROR SAVING IMAGE: << Unsupported format >> \n\n");
}
}
void init_centers(byte_t *data, double *centers, int n_px, int n_ch, int n_clus)
{
int k, ch, rnd;
#pragma omp parallel
{
srand((int)(seed) ^ omp_get_thread_num());
#pragma omp for private(k, ch, rnd)
for (k = 0; k < n_clus; k++) {
rnd = rand() % n_px;
for (ch = 0; ch < n_ch; ch++) {
centers[k * n_ch + ch] = data[rnd * n_ch + ch];
}
}
}
}
void label_pixels(byte_t *data, double *centers, int *labels, double *dists, int *changes, int n_px, int n_ch, int n_clus)
{
int px, ch, k;
int min_k, tmp_changes = 0;
double dist, min_dist, tmp;
#pragma omp parallel for schedule(guided, 100) private(px, ch, k, min_k, dist, min_dist, tmp)
for (px = 0; px < n_px; px++) {
min_dist = DBL_MAX;
for (k = 0; k < n_clus; k++) {
dist = 0;
for (ch = 0; ch < n_ch; ch++) {
tmp = (double)(data[px * n_ch + ch] - centers[k * n_ch + ch]);
dist += tmp * tmp;
}
if (dist < min_dist) {
min_dist = dist;
min_k = k;
}
}
dists[px] = min_dist;
if (labels[px] != min_k) {
labels[px] = min_k;
tmp_changes = 1;
}
}
*changes = tmp_changes;
}
void update_centers(byte_t *data, double *centers, int *labels, double *dists, int n_px, int n_ch, int n_clus)
{
int px, ch, k;
int *counts;
int min_k, far_px;
double max_dist;
counts = malloc(n_clus * sizeof(int));
for (k = 0; k < n_clus; k++) {
for (ch = 0; ch < n_ch; ch++) {
centers[k * n_ch + ch] = 0;
}
counts[k] = 0;
}
#pragma omp parallel for private(px, ch, min_k) reduction(+:centers[:n_clus * n_ch],counts[:n_clus])
for (px = 0; px < n_px; px++) {
min_k = labels[px];
for (ch = 0; ch < n_ch; ch++) {
centers[min_k * n_ch + ch] += data[px * n_ch + ch];
}
counts[min_k]++;
}
#pragma omp parallel for private(px, ch, min_k, k)
for (k = 0; k < n_clus; k++) {
if (counts[k]) {
for (ch = 0; ch < n_ch; ch++) {
centers[k * n_ch + ch] /= counts[k];
}
} else {
max_dist = 0;
for (px = 0; px < n_px; px++) {
if (dists[px] > max_dist) {
max_dist = dists[px];
far_px = px;
}
}
for (ch = 0; ch < n_ch; ch++) {
centers[k * n_ch + ch] = data[far_px * n_ch + ch];
}
dists[far_px] = 0;
}
}
free(counts);
}
void update_image(byte_t *data, double *centers, int *labels, int n_px, int n_ch)
{
int px, ch, min_k;
for (px = 0; px < n_px; px++) {
min_k = labels[px];
for (ch = 0; ch < n_ch; ch++) {
data[px * n_ch + ch] = (byte_t)round(centers[min_k * n_ch + ch]);
}
}
}
void kmeans(byte_t *data, int width, int height, int n_channels, int n_clus, int *n_iters, int n_threads)
{
int n_px;
int iter, max_iters;
int *changes;
int *labels;
double *centers;
double *dists;
omp_set_num_threads(n_threads);
max_iters = *n_iters;
n_px = width * height;
int n_px2 = n_px / 2;
labels = malloc(n_px * sizeof(int));
centers = malloc(n_clus * n_channels * sizeof(double));
dists = malloc(n_px * sizeof(double));
byte_t *data1 = data;
byte_t *data2 = data + n_px2 * n_channels;
int *labels1 = labels;
int *labels2 = labels + n_px2;
double *dists1 = dists;
double *dists2 = dists + n_px2;
init_centers(data, centers, n_px, n_channels, n_clus);
FILE *fp;
char *source_str;
size_t source_size;
fp = fopen("kernel.cl", "r");
if (!fp) {
fprintf(stderr, ":-(#\n");
exit(1);
}
source_str = (char*)malloc(MAX_SOURCE_SIZE);
source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
source_str[source_size] = '\0';
fclose( fp );
size_t local_item_size = WORKGROUP_SIZE;
size_t num_groups = n_px2 * local_item_size;
size_t global_item_size = num_groups * local_item_size;
cl_int ret;
cl_platform_id	platform_id[10];
cl_uint			ret_num_platforms;
ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
cl_device_id	device_id[10];
cl_uint			ret_num_devices;
ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 2, device_id, &ret_num_devices);
cl_context context = clCreateContext(NULL, 2, device_id, NULL, NULL, &ret);
cl_command_queue command_queue1 = clCreateCommandQueue(context, device_id[0], 0, &ret);
cl_command_queue command_queue2 = clCreateCommandQueue(context, device_id[1], 0, &ret);
cl_mem gpu_data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px2 * n_channels, data1, &ret);
cl_mem gpu_centers = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, n_clus * n_channels * sizeof(double), centers, &ret);
cl_mem gpu_labels = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px2 * sizeof(int), labels1, &ret);
cl_mem gpu_dists = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px2 * sizeof(double), dists1, &ret);
cl_mem gpu_data2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px2 * n_channels, data2, &ret);
cl_mem gpu_centers2 = clCreateBuffer(context, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, n_clus * n_channels * sizeof(double), centers, &ret);
cl_mem gpu_labels2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px2 * sizeof(int), labels2, &ret);
cl_mem gpu_dists2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n_px2 * sizeof(double), dists2, &ret);
cl_program program1 = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &ret);
ret = clBuildProgram(program, 2, device_id, NULL, NULL, NULL);
size_t build_log_len;
char *build_log;
ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
build_log =(char *)malloc(sizeof(char)*(build_log_len+1));
ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
printf("%s\n", build_log);
free(build_log);
cl_kernel kernel1 = clCreateKernel(program, "labelPixels", &ret);
cl_kernel kernel2 = clCreateKernel(program, "labelPixels2", &ret);
size_t buf_size_t;
clGetKernelWorkGroupInfo(kernel1, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof(buf_size_t), &buf_size_t, NULL);
ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&gpu_data);
ret |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&gpu_centers);
ret |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void*)&gpu_labels);
ret |= clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void*)&gpu_dists);
ret |= clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*)&n_px2);
ret |= clSetKernelArg(kernel1, 5, sizeof(cl_int), (void*)&n_channels);
ret |= clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*)&n_clus);
ret = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&gpu_data2);
ret |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&gpu_centers2);
ret |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void*)&gpu_labels2);
ret |= clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void*)&gpu_dists2);
ret |= clSetKernelArg(kernel2, 4, sizeof(cl_int), (void*)&n_px2);
ret |= clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*)&n_channels);
ret |= clSetKernelArg(kernel2, 6, sizeof(cl_int), (void*)&n_clus);
double labeling = 0;
double updating = 0;
printf("Training...\n");
for (iter = 0; iter < max_iters; iter++) {
double dt;
dt = omp_get_wtime();
ret = clEnqueueNDRangeKernel(command_queue1, kernel1, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
ret = clEnqueueNDRangeKernel(command_queue2, kernel2, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
ret = clEnqueueReadBuffer(command_queue1, gpu_data, CL_TRUE, 0, n_px2 * n_channels, data1, 0, NULL, NULL);
ret = clEnqueueReadBuffer(command_queue1, gpu_labels, CL_TRUE, 0, n_px2 * sizeof(int), labels1, 0, NULL, NULL);
ret = clEnqueueReadBuffer(command_queue1, gpu_dists, CL_TRUE, 0, n_px2 * sizeof(double), dists1, 0, NULL, NULL);
ret = clEnqueueReadBuffer(command_queue2, gpu_data2, CL_TRUE, 0, n_px2 * n_channels, data2, 0, NULL, NULL);
ret = clEnqueueReadBuffer(command_queue2, gpu_labels2, CL_TRUE, 0, n_px2 * sizeof(int), labels2, 0, NULL, NULL);
ret = clEnqueueReadBuffer(command_queue2, gpu_dists2, CL_TRUE, 0, n_px2 * sizeof(double), dists2, 0, NULL, NULL);
labeling += omp_get_wtime() - dt;
dt = omp_get_wtime();
update_centers(data, centers, labels, dists, n_px, n_channels, n_clus);
updating += omp_get_wtime() - dt;
}
update_image(data, centers, labels, n_px, n_channels);
*n_iters = iter;
printf("Labeling %f\nUpdating %f\n", labeling, updating);
ret = clFlush(command_queue1);
ret = clFinish(command_queue1);
ret = clFlush(command_queue2);
ret = clFinish(command_queue2);
ret = clReleaseKernel(kernel1);
ret = clReleaseKernel(kernel2);
ret = clReleaseProgram(program);
ret = clReleaseMemObject(gpu_data);
ret = clReleaseMemObject(gpu_labels);
ret = clReleaseMemObject(gpu_centers);
ret = clReleaseMemObject(gpu_dists);
ret = clReleaseMemObject(gpu_data2);
ret = clReleaseMemObject(gpu_labels2);
ret = clReleaseMemObject(gpu_centers2);
ret = clReleaseMemObject(gpu_dists2);
ret = clReleaseCommandQueue(command_queue1);
ret = clReleaseCommandQueue(command_queue2);
ret = clReleaseContext(context);
free(centers);
free(labels);
free(dists);
}
void print_usage(char *pgr_name)
{
char *usage = "\nPROGRAM USAGE \n\n"
"   %s [-h] [-k num_clusters] [-i max_iters] [-o output_img] [-s seed] input_image \n\n"
"OPTIONAL PARAMETERS \n\n"
"   -k num_clusters : number of clusters, must be bigger than 1. Default is %d. \n"
"   -i max_iters    : Max Number of iterations, must be bigger than 0. defualt is %d \n"
"   -o output_image : output image path, include extenstion. \n"
"   -s seed         : seed for random function. \n"
"   -t threads      : number of threads for OpenMP\n"
"   -h              : print help. \n\n";
fprintf(stderr, usage, pgr_name, 4, 150);
}
void print_exec(int width, int height, int n_ch, int n_clus, int n_iters, off_t inSize, off_t outSize, double dt)
{
char *details = "\nEXECUTION\n\n"
"  Image size              : %d x %d\n"
"  Color channels          : %d\n"
"  Number of clusters      : %d\n"
"  Number of iterations    : %d\n"
"  Input Image Size        : %ld KB\n"
"  Output Image Size       : %ld KB\n"
"  Size diffrance          : %ld KB\n"
"  Runtime                 : %f\n\n";
fprintf(stdout, details, width, height, n_ch, n_clus, n_iters, inSize / 1000, outSize / 1000, (inSize - outSize) / 1000, dt);
}
off_t fsize(const char *filename)
{ 
struct stat st;
if (stat(filename, &st) == 0)
return st.st_size;
fprintf(stderr, "Cannot determine size of %s: %s\n",
filename, strerror(errno));
return -1;
}
int main(int argc, char **argv)
{
int n_iters = 150;
int n_clus = 4;
char *out_path = "result.png";
int width, height, n_ch, n_threads = N_THREADS;
byte_t *data;
seed = time(NULL);
char *in_path = NULL;
char optchar;
while ((optchar = getopt(argc, argv, "k:i:o:s:t:h")) != -1) {
switch (optchar) {
case 'k':
n_clus = strtol(optarg, NULL, 10);
break;
case 'i':
n_iters = strtol(optarg, NULL, 10);
break;
case 'o':
out_path = optarg;
break;
case 's':
seed = strtol(optarg, NULL, 10);
break;
case 't':
n_threads = strtol(optarg, NULL, 10);
break;
case 'h':
default:
print_usage(argv[0]);
exit(EXIT_FAILURE);
break;
}
}
in_path = argv[optind];
if (in_path == NULL) {
print_usage(argv[0]);
exit(EXIT_FAILURE);
}
if (n_clus < 2) {
fprintf(stderr, "INPUT ERROR: << Invalid number of clusters >> \n");
exit(EXIT_FAILURE);
}
if (n_iters < 1) {
fprintf(stderr, "INPUT ERROR: << Invalid maximum number of iterations >> \n");
exit(EXIT_FAILURE);
}
data = img_load(in_path, &width, &height, &n_ch);
double dt = omp_get_wtime();
kmeans(data, width, height, n_ch, n_clus, &n_iters, n_threads);
dt = omp_get_wtime() - dt;
img_save(out_path, data, width, height, n_ch);
off_t inSize = fsize(in_path);
off_t outSize = fsize(out_path);
print_exec(width, height, n_ch, n_clus, n_iters, inSize, outSize, dt);
free(data);
return EXIT_SUCCESS;
}
