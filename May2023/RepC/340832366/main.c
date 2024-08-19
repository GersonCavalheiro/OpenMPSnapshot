#include <malloc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "MyUtils.h"
int n_threads = 1;
#define THREAD_NUM_ARG 1
#define DIR_ARG 2
#define DICT_ARG 3
#define RES_ARG 4
#define DICT_FILE_BUFFER_SIZE 1000000
char dict_file_buffer[DICT_FILE_BUFFER_SIZE];
int main(int argc, char *argv[]) {
int dict_size;     
int file_count;    
char **filenames;  
int **vectors;     
if (argc != 5) {
printf("程序需要输入4个参数，用法如下：\n");
printf("%s <n_threads> <dir> <dict> <results>\n", argv[0]);
exit(-1);
}
n_threads = atoi(argv[THREAD_NUM_ARG]);
omp_set_num_threads(n_threads);
double ts = omp_get_wtime();
readAll(dict_file_buffer, argv[DICT_ARG]);
dict_size = make_dict_Hash(dict_file_buffer);
file_count = get_names(argv[DIR_ARG], &filenames);
vectors = (int **)calloc(file_count, sizeof(int *));
#pragma omp parallel for
for (int i = 0; i < file_count; ++i) {
vectors[i] = (int *)calloc(dict_size, sizeof(int));
make_profile(filenames[i], dict_size, vectors[i]);
}
write_profiles(argv[RES_ARG], file_count, dict_size, filenames, vectors);
double te = omp_get_wtime();
printf("Time:%f s\n", te - ts);
}