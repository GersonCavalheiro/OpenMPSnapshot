#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include <time.h>
#include "util/queue.h"
#include "util/hashTable.h"
#include "util/util.h"
#define NUM_THREADS 16
#define REPEAT_FILES 10
#define HASH_CAPACITY 50000
extern int errno;
int DEBUG_MODE = 0;
int PRINT_MODE = 1;
int main(int argc, char **argv)
{
char files_dir[] = "./files"; 
omp_lock_t readlock;
omp_init_lock(&readlock);
double time = -omp_get_wtime();
int file_count = 0;
struct Queue *file_name_queue;
file_name_queue = createQueue();
for (int i = 0; i < REPEAT_FILES; i++)
{
int files = get_file_list(file_name_queue, files_dir);
if (files == -1)
{
printf("Check input directory and rerun! Exiting!\n");
return 1;
}
file_count += files;
}
printf("file_count %d\n", file_count);
struct Queue **queues;
struct hashtable **hash_tables;
queues = (struct Queue **)malloc(sizeof(struct Queue *) * NUM_THREADS/2);
hash_tables = (struct hashtable **)malloc(sizeof(struct hashtable *) * NUM_THREADS/2);
omp_lock_t queuelock[NUM_THREADS/2];
for (int i = 0; i < NUM_THREADS/2; i++)
{
omp_init_lock(&queuelock[i]);
queues[i] = createQueue();
}
omp_set_num_threads(NUM_THREADS);
int i;
#pragma omp parallel default(none) shared(queues, file_name_queue, hash_tables, readlock, queuelock)
{
int threadn = omp_get_thread_num();
if (threadn < NUM_THREADS/2) 
{
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
int thread = threadn - NUM_THREADS/2;
hash_tables[thread] = createtable(50000);
populateHashMapWL(queues[thread], hash_tables[thread], &queuelock[thread]);
}
}
printf("destroying the lock\n");
omp_destroy_lock(&readlock);
for (int k=0; k<NUM_THREADS/2; k++) {
omp_destroy_lock(&queuelock[k]);
}
printf("reading and mapping done\n");
struct hashtable *final_table = createtable(HASH_CAPACITY);
#pragma omp parallel shared(final_table, hash_tables)
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
reduce(hash_tables, final_table, NUM_THREADS/2, i);
}
}
printf("reduction done\n");
#pragma omp parallel shared(final_table)
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
char *filename = (char *)malloc(sizeof(char) * 30);
sprintf(filename, "output/parallel/%d.txt", threadn);
writePartialTable(final_table, filename, start, end);
}
#pragma omp parallel for
for (i = 0; i < NUM_THREADS/2; i++)
{
free(queues[i]);
free(hash_tables[i]);
}
free(queues);
free(hash_tables);
time += omp_get_wtime();
printf("total time taken for the execution: %f\n", time);
return EXIT_SUCCESS;
}
