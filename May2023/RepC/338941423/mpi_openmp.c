#include <dirent.h>
#include <errno.h>
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>
#include "util/hashTable.h"
#include "util/queue.h"
#include "util/util.h"
#define TAG_COMM_REQ_DATA 0
#define TAG_COMM_FILE_NAME 1
#define TAG_COMM_PAIR_LIST 3
#define WORD_MAX_LENGTH 50
#define HASH_CAPACITY 50000
#define NUM_THREADS 4
#define REPEAT_FILES 10
#define BREAK_RM 0
extern int errno;
int DEBUG_MODE = 0;
int PRINT_MODE = 1;
typedef struct
{
int hash;
int count;
char word[WORD_MAX_LENGTH];
} pair;
int args_parse(int argc, char **argv, char *files_dir, int *repeat_files, int *nthreads, int *par_read_map)
{
int opt;
while ((opt = getopt(argc, argv, "d:r:t:b:g")) != -1)
{
switch (opt)
{
case 'd':
strcpy(files_dir, optarg);
break;
case 'r':
*repeat_files = (int)atol(optarg);
break;
case 't':
*nthreads = atoi(optarg);
break;
case 'b':
*par_read_map = atoi(optarg);
break;
case ':':
fprintf(stderr, "Option -%c requires an argument to be given\n", optopt);
return -1;
}
}
return 0;
}
int main(int argc, char **argv)
{
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
if(provided < MPI_THREAD_FUNNELED)
{
fprintf(stderr, "Error: the threading support level: %d is lesser than that demanded: %d\n", MPI_THREAD_FUNNELED, provided);
MPI_Finalize();
return 0;
}
int size, pid, p_name_len;
char p_name[MPI_MAX_PROCESSOR_NAME];
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &pid);
MPI_Get_processor_name(p_name, &p_name_len);
int par_read_map = BREAK_RM;
int nthreads = NUM_THREADS;
char files_dir[] = "./files";
int repeat_files = REPEAT_FILES;
double global_time = 0;
double local_time = 0;
char csv_out[400] = "";
char tmp_out[200] = "";
int arg_parse = args_parse(argc, argv, files_dir, &repeat_files, &nthreads, &par_read_map);
if(par_read_map ==1 && nthreads<2 && nthreads%2!=0) {
fprintf(stderr, "Error: number of threads cannot be a odd or below 2\n");
MPI_Finalize();
return 1;
}
FILE *outfile;
char buf[16];
snprintf(buf, 16, "./pout/f%03d.txt", pid);
outfile = fopen(buf, "w");
MPI_Request request;
MPI_Status status;
int recv_pid;
int recv_len = 0;
int count = 0;
int done = 0;
int file_count = 0;
int done_sent_p_count = 0;
struct Queue *file_name_queue;
file_name_queue = createQueue();
struct Queue *files_to_read;
MPI_Barrier(MPI_COMM_WORLD);
local_time = -omp_get_wtime();
char *file_names = (char *)malloc(sizeof(char) * FILE_NAME_BUF_SIZE * 1000);
if (pid == 0)
{
for (int i = 0; i < repeat_files; i++)
{
int files = get_file_list(file_name_queue, files_dir);
if (files == -1)
{
printf("Check input directory and rerun! Exiting!\n");
return 1;
}
file_count += files;
}
int num_files_to_send = file_count / size;
int spare = file_count % size;
for (int i = 0; i < size; i++)
{
char *concat_files =
(char *)malloc(sizeof(char) * FILE_NAME_BUF_SIZE * num_files_to_send);
int len = 0;
int send_file_count = num_files_to_send;
if (spare>0) {
send_file_count += 1;
spare--;
}
for (int j = 0; j < send_file_count; j++)
{
if (j == 0)
{
strcpy(concat_files, file_name_queue->front->line);
}
else
{
strcat(concat_files, file_name_queue->front->line);
}
len += file_name_queue->front->len + 1;
if (j != send_file_count - 1)
{
strcat(concat_files, ",");
}
deQueue(file_name_queue);
}
if (i == 0)
{ 
strcpy(file_names, concat_files);
recv_len = len;
}
else
{
MPI_Send(concat_files, len + 1, MPI_CHAR, i, TAG_COMM_FILE_NAME,
MPI_COMM_WORLD);
}
}
}
else
{
MPI_Status status;
MPI_Recv(file_names, FILE_NAME_BUF_SIZE * 1000, MPI_CHAR, 0,
TAG_COMM_FILE_NAME, MPI_COMM_WORLD, &status);
MPI_Get_count(&status, MPI_CHAR, &recv_len);
}
char *file;
int received_file_count = 0;
while ((file = strtok_r(file_names, ",", &file_names)))
{
if (strlen(file) > 0)
{
enQueue(file_name_queue, file, strlen(file));
received_file_count++;
}
}
MPI_Barrier(MPI_COMM_WORLD);
local_time += omp_get_wtime();
global_time += local_time;
sprintf(tmp_out, "%d, %d, %d, %d, %d, %.4f, ", file_count, HASH_CAPACITY, size, nthreads, par_read_map, local_time);
strcat(csv_out, tmp_out);   
local_time = -omp_get_wtime();
omp_lock_t readlock;
omp_init_lock(&readlock);
omp_lock_t queuelock[nthreads/2];
struct Queue **queues;
struct hashtable **hash_tables;
queues = (struct Queue **)malloc(sizeof(struct Queue *) * nthreads);
hash_tables = (struct hashtable **)malloc(sizeof(struct hashtable *) * nthreads);
for (int k=0; k<nthreads; k++) {
omp_init_lock(&queuelock[k]);
queues[k] = createQueue();
hash_tables[k] = createtable(HASH_CAPACITY);
}
#pragma omp parallel shared(queues, hash_tables, file_name_queue, readlock, queuelock) num_threads(nthreads)
{
int threadn = omp_get_thread_num();
if (par_read_map) {
if (threadn < nthreads/2) {
while (file_name_queue->front != NULL)
{
char file_name[30];
omp_set_lock(&readlock);
if (file_name_queue->front == NULL) {
omp_unset_lock(&readlock);
continue;
}
strcpy(file_name, file_name_queue->front->line);
deQueue(file_name_queue);
omp_unset_lock(&readlock);
populateQueueWL(queues[threadn], file_name, &queuelock[threadn]);
}
queues[threadn]->finished = 1;
} else {
int thread = threadn - nthreads/2;
hash_tables[thread] = createtable(HASH_CAPACITY);
populateHashMapWL(queues[thread], hash_tables[thread], &queuelock[thread]);
}
} else {
while (file_name_queue->front != NULL)
{
char file_name[30];
omp_set_lock(&readlock);
if (file_name_queue->front == NULL) {
omp_unset_lock(&readlock);
continue;
}
strcpy(file_name, file_name_queue->front->line);
deQueue(file_name_queue);
omp_unset_lock(&readlock);
queues[threadn]->finished = 0;
populateQueue(queues[threadn], file_name);         
queues[threadn]->finished = 1;
populateHashMap(queues[threadn], hash_tables[threadn]); 
}
}
}
omp_destroy_lock(&readlock);
for (int k=0; k<nthreads; k++) {
omp_destroy_lock(&queuelock[k]);
}
MPI_Barrier(MPI_COMM_WORLD);
local_time += omp_get_wtime();
global_time += local_time;
sprintf(tmp_out, "%.4f, ", local_time);
strcat(csv_out, tmp_out);
local_time = -omp_get_wtime();
struct hashtable *final_table = createtable(HASH_CAPACITY);
#pragma omp parallel shared(final_table, hash_tables) num_threads(nthreads)
{
int threadn = omp_get_thread_num();
int tot_threads = omp_get_num_threads();
int interval = HASH_CAPACITY / tot_threads;
int start = threadn * interval;
int end = start + interval;
if (end > final_table->tablesize)
{
end = final_table->tablesize;
}
int i;
for (i = start; i < end; i++)
{
reduce(hash_tables, final_table, nthreads, i);
}
}
MPI_Barrier(MPI_COMM_WORLD);
local_time += omp_get_wtime();
global_time += local_time;
sprintf(tmp_out, "%.4f, ", local_time);
strcat(csv_out, tmp_out);
local_time = -omp_get_wtime();
int h_space = HASH_CAPACITY / size;
int h_start = h_space * pid;
int h_end = h_space * (pid + 1);
const int nfields = 3;
MPI_Aint disps[nfields];
int blocklens[] = {1, 1, WORD_MAX_LENGTH};
MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_CHAR};
disps[0] = offsetof(pair, hash);
disps[1] = offsetof(pair, count);
disps[2] = offsetof(pair, word);
MPI_Datatype istruct;
MPI_Type_create_struct(nfields, blocklens, disps, types, &istruct);
MPI_Type_commit(&istruct);
int k = 0;
for (k = 0; k < size; k++)
{
if (pid != k)
{
int j = 0;
pair pairs[HASH_CAPACITY];
struct node *current = NULL;
for (int i = h_space * k; i < h_space * (k + 1); i++)
{
current = final_table->table[i];
if (current == NULL)
continue;
while (current != NULL)
{
pairs[j].count = current->frequency;
pairs[j].hash = i;
strcpy(pairs[j].word, current->key);
j++;
current = current->next;
}
}
MPI_Send(pairs, j, istruct, k, TAG_COMM_PAIR_LIST, MPI_COMM_WORLD);
}
else if (pid == k)
{
for (int pr = 0; pr < size - 1; pr++)
{
int recv_j = 0;
pair recv_pairs[HASH_CAPACITY];
MPI_Recv(recv_pairs, HASH_CAPACITY, istruct, MPI_ANY_SOURCE,
TAG_COMM_PAIR_LIST, MPI_COMM_WORLD, &status);
MPI_Get_count(&status, istruct, &recv_j);
for (int i = 0; i < recv_j; i++)
{
pair recv_pair = recv_pairs[i];
int frequency = recv_pair.count;
struct node *node = add(final_table, recv_pair.word, 0);
node->frequency += recv_pair.count;
}
}
}
}
MPI_Barrier(MPI_COMM_WORLD);
local_time += omp_get_wtime();
global_time += local_time;
sprintf(tmp_out, "%.4f, ", local_time);
strcat(csv_out, tmp_out);
local_time = -omp_get_wtime();
writeTable(final_table, outfile, h_start, h_end);
local_time += omp_get_wtime();
global_time += local_time;
sprintf(tmp_out, "%.4f, ", local_time);
strcat(csv_out, tmp_out);
sprintf(tmp_out, "%.4f", global_time);
strcat(csv_out, tmp_out);
if (pid == 0) {
fprintf(stdout, "Num_Files, Hash_size, Num_Processes, Num_Threads, Par_Read_Map, FS_Time, Read_Map, Local_Reduce, Final_Reduce, Write, Total\n%s\n\n", 
csv_out);
}
MPI_Finalize();
return 0;
}