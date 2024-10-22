#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include <time.h>
#include "util/queue.h"
#include "util/hashTable.h"
#include "util/util.h"
#define NUM_THREADS 8
#define REPEAT_FILES 10
#define HASH_CAPACITY 50000
extern int errno;
int DEBUG_MODE = 0;
int PRINT_MODE = 1;
void populateHashMap2(struct Queue *q, struct hashtable *hashMap)
{
struct node *node = NULL;
while (q == NULL)
{
delay(5);
continue;
}
int count = 0;
while (q->front != NULL || !q->finished)
{
if (q->front == NULL)
{
printf("map waiting for data ..\n");
delay(5);
continue;
}
count++;
if (count % 10000 == 0)
{
printf(".");
}
char str[q->front->len];
strcpy(str, q->front->line);
char *token;
char *rest = str;
while ((token = strtok_r(rest, " ", &rest)))
{
char *word = format_string(token);
if (strlen(word) > 0)
{
node = add(hashMap, word, 0);
node->frequency++;
}
free(word);
}
deQueue(q);
}
}
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
queues = (struct Queue **)malloc(sizeof(struct Queue *) * NUM_THREADS);
hash_tables = (struct hashtable **)malloc(sizeof(struct hashtable *) * NUM_THREADS);
for (int i = 0; i < NUM_THREADS; i++)
{
queues[i] = createQueue();
}
int i;
omp_set_nested(1); 
#pragma omp parallel default(none) shared(queues, file_name_queue, hash_tables, readlock) num_threads(2)
#pragma omp single
{
#pragma omp task 
{
#pragma omp parallel for num_threads(NUM_THREADS) shared(queues, file_name_queue, readlock)
for (int i = 0; i < NUM_THREADS; i++)
{ 
int threadn = omp_get_thread_num();
printf("read section thread %d, i %d\n", threadn, i);
while (file_name_queue->front != NULL)
{
char file_name[30];
omp_set_lock(&readlock);
strcpy(file_name, file_name_queue->front->line);
deQueue(file_name_queue);
omp_unset_lock(&readlock);
populateQueue(queues[i], file_name);
}
queues[i]->finished = 1;
}
}
#pragma omp task 
{
#pragma omp parallel for num_threads(NUM_THREADS) shared(queues, hash_tables)
for (int i = 0; i < NUM_THREADS; i++)
{
int threadn = omp_get_thread_num();
hash_tables[i] = createtable(50000);
printf("map section thread %d, i %d\n", threadn, i);
populateHashMap2(queues[i], hash_tables[i]);
}
}
}
printf("destroying the lock\n");
omp_destroy_lock(&readlock);
struct hashtable *final_table = createtable(HASH_CAPACITY);
#pragma omp parallel for shared(final_table, hash_tables) num_threads(NUM_THREADS)
for (int i = 0; i < NUM_THREADS; i++)
{
int threadn = omp_get_thread_num();
int tot_threads = omp_get_num_threads();
int interval = HASH_CAPACITY / tot_threads;
int start = threadn * interval;
int end = start + interval;
printf("reduce section thread %d, start %d\n", threadn, start);
if (end > final_table->tablesize)
{
end = final_table->tablesize;
}
int i;
for (i = start; i < end; i++)
{
reduce(hash_tables, final_table, NUM_THREADS, i);
}
}
#pragma omp parallel for num_threads(NUM_THREADS) shared(final_table)
for (int i = 0; i < NUM_THREADS; i++)
{
int threadn = omp_get_thread_num();
int tot_threads = omp_get_num_threads();
int interval = HASH_CAPACITY / tot_threads;
int start = threadn * interval;
int end = start + interval;
if (end > hash_tables[threadn]->tablesize)
{
end = final_table->tablesize;
}
printf("write thread %d\n", threadn);
char *filename = (char *)malloc(sizeof(char) * 30);
sprintf(filename, "output/parallel/%d.txt", threadn);
writePartialTable(final_table, filename, start, end);
}
#pragma omp parallel for
for (i = 0; i < NUM_THREADS; i++)
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
