#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>
#include <string.h>
#include <inttypes.h>
#include <stdint.h>
#include <omp.h>
#define MAXBUF 1000
#define MAXCPU 100
#define MAXK 5000
#define MAXLOG 5000
#define MAXN_POWER 30
#define CHUNK_POWER 10  
#define CHUNK_SIZE (1 << (CHUNK_POWER)) 
#define MAXGROUP (1 << (MAXN_POWER-CHUNK_POWER))
#define GROUP_STRIDE 128 
#define MAXGROUP_STRIDE_NUM (MAXGROUP / GROUP_STRIDE)
#define DEVICE_NUM 1 
#define TASK_DEVICE_NUM 2 
cl_int status;
cl_uint array[TASK_DEVICE_NUM][MAXGROUP_STRIDE_NUM];
typedef struct Keys_{
cl_uint key1[1];
cl_uint key2[1];
} Keys;
typedef struct InputRecord_{
cl_int N;
Keys key_pair;
} InputRecord;
static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
return  (x << n) | (x >> (32-n));
}
static inline uint32_t encrypt(uint32_t m, uint32_t key) {
return (rotate_left(m, key&31) + key)^key;
}
void platform(cl_platform_id *platform_id, cl_uint *platform_id_got){
status = clGetPlatformIDs(1, platform_id, 
platform_id_got);
assert(status == CL_SUCCESS && *platform_id_got == 1);
#ifdef DEBUG
printf("%d platform found\n", *platform_id_got);
#endif
}
void getDevice(cl_platform_id *platform_id, cl_device_id CPU[], cl_uint *CPU_id_got){
#ifndef NOTUSEGPU
status = clGetDeviceIDs(*platform_id, CL_DEVICE_TYPE_GPU, 
MAXCPU, CPU, CPU_id_got);
#else
status = clGetDeviceIDs(*platform_id, CL_DEVICE_TYPE_CPU, 
MAXCPU, CPU, CPU_id_got);
#endif
assert(status == CL_SUCCESS);
#ifdef DEBUG
printf("There are %d CPU devices\n", *CPU_id_got); 
#endif
}
void getContext(cl_uint *CPU_id_got, cl_device_id CPU[], cl_context *context){
*context = clCreateContext(NULL, TASK_DEVICE_NUM, CPU, NULL, NULL, &status);
assert(status == CL_SUCCESS);
#ifdef DEBUG
printf("Context Success\n");
#endif
}
void createCommandQueue(cl_context *context, cl_device_id *device_id, cl_command_queue *commandQueue){
*commandQueue = clCreateCommandQueue(*context, *device_id, 0, &status);
assert(status == CL_SUCCESS);
#ifdef DEBUG
printf("Command Queue Success");
#endif
}
void load_my_code(char *filename, cl_program *program, cl_context *context){
FILE *kernelfp = fopen(filename, "r");
assert(kernelfp != NULL);
char kernelBuffer[MAXK];
const char *constKernelSource = kernelBuffer;
size_t kernelLength = fread(kernelBuffer, 1, MAXK, kernelfp);
*program = 
clCreateProgramWithSource(*context, 1, &constKernelSource, 
&kernelLength, &status);
assert(status == CL_SUCCESS);
#ifdef DEBUG
printf("Create Program Success\n");
#endif
}
void buildProgram(cl_program *program, cl_uint *CPU_id_got, cl_device_id CPU[]){
assert(TASK_DEVICE_NUM<= *CPU_id_got);
status = 
clBuildProgram(*program, TASK_DEVICE_NUM, CPU, NULL, NULL, 
NULL);
}
void getbuildProgramInfo(cl_program *program, cl_device_id *device_id, char logInformation[], size_t logInfoByteSize){
size_t return_length;
status = clGetProgramBuildInfo(*program, *device_id, CL_PROGRAM_BUILD_LOG, logInfoByteSize, logInformation, &return_length);
logInformation[return_length/sizeof(logInformation[0])] = '\0';
}
void buildKernel(cl_program *program, const char *function_name, cl_kernel *kernel){
*kernel = clCreateKernel(*program, function_name, &status);
assert(status == CL_SUCCESS);
}
void buildBuffers(cl_context *context, cl_mem *arrayBuffer, cl_uint array[]){
*arrayBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, MAXGROUP_STRIDE_NUM * sizeof(array[0]), array, &status);
assert(status == CL_SUCCESS);
}
void static_link_params(cl_kernel *kernel, cl_mem *arrayBuffer){
status = clSetKernelArg(*kernel, 2, sizeof(cl_uint) * CHUNK_SIZE, NULL); 
assert(status == CL_SUCCESS);
status = clSetKernelArg(*kernel, 3, sizeof(cl_mem), (void *)arrayBuffer);
assert(status == CL_SUCCESS);
}
void dynamic_link_params(cl_kernel *kernel, Keys *keys, cl_int *N){
status = clSetKernelArg(*kernel, 0, sizeof(keys->key1[0]), (void *)&keys->key1[0]);
assert(status == CL_SUCCESS);
status = clSetKernelArg(*kernel, 1, sizeof(keys->key2[0]), (void *)&keys->key2[0]);
assert(status == CL_SUCCESS);
status = clSetKernelArg(*kernel, 4, sizeof(N[0]), (void *)&N[0]);
assert(status == CL_SUCCESS);
}
uint32_t enqueueCommand(Keys *keys, cl_command_queue commandQueue[DEVICE_NUM], cl_kernel kernel[DEVICE_NUM], int dim, size_t *global_work_size, size_t *local_work_size, size_t *group_number, cl_int N[], cl_int group_offsets[DEVICE_NUM], int *group_stride_workload){
for(int i = 0; i < dim; i++){
if(N[i] % local_work_size[i] != 0){
group_number[i] = (N[i] / local_work_size[i]) + 1;
}
else group_number[i] = N[i] / local_work_size[i];
global_work_size[i] = group_number[i] * local_work_size[i];
#ifdef DEBUG
printf("%lu\n", group_number[i]);
#endif
}
if(group_number[0] % GROUP_STRIDE != 0){
int num_pad_group = GROUP_STRIDE - (group_number[0] % GROUP_STRIDE);
group_number[0] += num_pad_group; 
global_work_size[0] = local_work_size[0] * group_number[0] / GROUP_STRIDE;
}else{
global_work_size[0] = local_work_size[0] * group_number[0] / GROUP_STRIDE;
}
int group_stride_num = (group_number[0] / GROUP_STRIDE); 
assert(group_number[0] % GROUP_STRIDE == 0);
*group_stride_workload = group_stride_num / DEVICE_NUM;
int group_stride_remainder = group_stride_num % DEVICE_NUM;
assert(group_stride_num - (*group_stride_workload * DEVICE_NUM) == group_stride_remainder);
for(int i = 0; i < DEVICE_NUM; i++){
group_offsets[i] = *group_stride_workload * i * GROUP_STRIDE; 
assert(group_offsets[i] % GROUP_STRIDE == 0);
if(i != DEVICE_NUM-1)
global_work_size[0] = local_work_size[0] * (*group_stride_workload);  
else
global_work_size[0] = local_work_size[0] * (*group_stride_workload + group_stride_remainder); 
status = clSetKernelArg(kernel[i], 5, sizeof(group_offsets[i]), (void *)&group_offsets[i]); 
assert(status == CL_SUCCESS);
status = clEnqueueNDRangeKernel(commandQueue[i], kernel[i], dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
assert(status == CL_SUCCESS);
}
uint32_t padsum = 0;
int end = local_work_size[0] * group_number[0]; 
for(int i = N[0]; i < end; i++){
padsum += encrypt(i, keys->key1[0]) * encrypt(i, keys->key2[0]);
}
return padsum;
}
void readResult(cl_command_queue commandQueue[DEVICE_NUM], cl_mem *arrayBuffer, cl_uint array[], size_t group_number[], cl_int group_offsets[DEVICE_NUM], int group_stride_workload, cl_event event[DEVICE_NUM]){
int group_stride_num = (group_number[0] / GROUP_STRIDE); 
assert(group_number[0] % GROUP_STRIDE == 0);
int group_stride_remainder = group_stride_num % DEVICE_NUM;
for(int i = 0; i < DEVICE_NUM; i++){
int array_offset_index = group_offsets[i]; 
size_t offset = sizeof(array[0]) * group_offsets[i];  
size_t bytes_being_read;
if(i != DEVICE_NUM - 1) bytes_being_read = sizeof(array[0]) * group_stride_workload; 
else{
bytes_being_read = sizeof(array[0]) * (group_stride_workload + group_stride_remainder); 
}
status = clEnqueueReadBuffer(commandQueue[i], *arrayBuffer, CL_TRUE, offset, bytes_being_read, array+array_offset_index, 0, NULL, NULL);
assert(status == CL_SUCCESS);
}
}
void updateBuffer(cl_command_queue *commandQueue, cl_mem *keybuffer1, cl_mem *keybuffer2, cl_mem *NBuffer, 
Keys *keys, cl_int N[]){
status = clEnqueueWriteBuffer(*commandQueue, *keybuffer1, CL_TRUE, 0, 1 * sizeof(keys->key1[0]), keys->key1, 0, NULL, NULL);
assert(status == CL_SUCCESS);
status = clEnqueueWriteBuffer(*commandQueue, *keybuffer2, CL_TRUE, 0, 1 * sizeof(keys->key2[0]), keys->key2, 0, NULL, NULL);
assert(status == CL_SUCCESS);
status = clEnqueueWriteBuffer(*commandQueue, *NBuffer, CL_TRUE, 0, 1 * sizeof(N[0]), N, 0, NULL, NULL);
assert(status == CL_SUCCESS);
}
void releaseAll(cl_uint array[TASK_DEVICE_NUM][MAXGROUP_STRIDE_NUM], cl_context *context, cl_command_queue commandQueue[TASK_DEVICE_NUM], 
cl_program *program, cl_kernel kernel[TASK_DEVICE_NUM], cl_mem arrayBuffer[TASK_DEVICE_NUM]){
clReleaseContext(*context);
for(int i = 0; i < TASK_DEVICE_NUM; i++){
clReleaseCommandQueue(commandQueue[i]);
}
clReleaseProgram(*program);
for(int i = 0; i < TASK_DEVICE_NUM; i++){
clReleaseKernel(kernel[i]);
clReleaseMemObject(arrayBuffer[i]);
}
}
int main(){
assert(DEVICE_NUM == 1); 
char filename[MAXBUF] = "vecdot.cl";
char logInformation[MAXLOG];
cl_platform_id platform_id;
cl_uint platform_id_got;
cl_device_id CPU[MAXCPU];
cl_uint CPU_id_got;
cl_context context;
cl_command_queue commandQueue[TASK_DEVICE_NUM];
cl_program program;
cl_kernel kernel[TASK_DEVICE_NUM];  
cl_mem arrayBuffer[TASK_DEVICE_NUM];
size_t global_work_size[TASK_DEVICE_NUM][1];
size_t local_work_size[TASK_DEVICE_NUM][1];
size_t group_number[TASK_DEVICE_NUM][1];
cl_int group_offsets[TASK_DEVICE_NUM];
int group_stride_workload[TASK_DEVICE_NUM];
cl_event event[TASK_DEVICE_NUM];
#ifdef DEBUG
#else
#endif
Keys keys[TASK_DEVICE_NUM];
cl_int N[TASK_DEVICE_NUM][1];
size_t len = strlen(filename);
assert(len > 0);
if(filename[len-1] == '\n') filename[len-1] = '\0';
#ifdef DEBUG
printf("filename: %s\n", filename);
#endif
platform(&platform_id, &platform_id_got);
getDevice(&platform_id, CPU, &CPU_id_got);
getContext(&CPU_id_got, CPU, &context);
for(int i = 0; i < TASK_DEVICE_NUM; i++){
createCommandQueue(&context, &CPU[i], &commandQueue[i]);
}
load_my_code(filename, &program, &context);
buildProgram(&program, &CPU_id_got, CPU);
#ifdef DEBUG
getbuildProgramInfo(&program, &CPU[0], logInformation, sizeof(logInformation));
printf("%s\n", logInformation);
#endif 
for(int i = 0; i < TASK_DEVICE_NUM; i++){
buildKernel(&program, "vecdot_load_balance", &kernel[i]);
}
for(int i = 0; i < TASK_DEVICE_NUM; i++){
buildBuffers(&context, &arrayBuffer[i], array[i]);
static_link_params(&kernel[i], &arrayBuffer[i]);
}
short has_task[TASK_DEVICE_NUM] = {0};
short done = 0;
uint32_t padsum[TASK_DEVICE_NUM] = {0};
while(1){
for(int i = 0; i < TASK_DEVICE_NUM && scanf("%d %" PRIu32 " %" PRIu32, &N[i][0], &keys[i].key1[0], &keys[i].key2[0]) == 3; i++){
local_work_size[i][0] = CHUNK_SIZE;
dynamic_link_params(&kernel[i], &keys[i], N[i]);
has_task[i] = 1;
padsum[i] = enqueueCommand(&keys[i], &commandQueue[i], &kernel[i], 1, global_work_size[i], local_work_size[i], group_number[i], N[i], &group_offsets[i], &group_stride_workload[i]);
}
for(int i = 0; i< TASK_DEVICE_NUM && has_task[i]; i++)
readResult(&commandQueue[i], &arrayBuffer[i], array[i], group_number[i], &group_offsets[i], group_stride_workload[i], &event[i]);
assert(status == CL_SUCCESS);
for(int i = 0; i < TASK_DEVICE_NUM; i++){
if(has_task[i] == 0){
done = 1;
break;
}
int group_num = group_number[i][0];
int group_stride_num = group_number[i][0] / GROUP_STRIDE; 
uint32_t sum = 0;
for(int j = 0; j < group_stride_num; j++){
sum += array[i][j];
#ifdef DEBUG
printf("array[%d]: %u\n", j, array[j]);
#endif
}
printf("%" PRIu32 "\n", sum - padsum[i]);
has_task[i] = 0;
}
if(done) break;
}
releaseAll(array, &context, commandQueue, &program, kernel, arrayBuffer);
return 0;
}
