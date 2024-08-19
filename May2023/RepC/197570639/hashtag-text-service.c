#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "hashtag-text-server.h"
extern char _start;
extern char _fini;
typedef struct __strings_node__ {
struct __strings_node__ *next;
struct __strings_node__ *prev;
char hashtags[HASHTAG_MAX_LENGTH];
char text[TEXT_MAX_LENGTH];
} strings_node;
typedef struct __strings_hash_table__ {
#ifdef NEW_NODES_INSERTION
omp_lock_t lock[STRINGS_HT_BUCKETS];
#endif
strings_node *list[STRINGS_HT_BUCKETS];
} strings_hash_table;
#if COMPLETED_REQUESTS_MONITORING
unsigned long long int completed_requests;
#endif
int number_requested_hashtags;
char requested_hashtags[MAX_DIFFERENT_REQUESTED_HASHTAGS][HASHTAG_MAX_LENGTH];
#ifdef NEW_NODES_INSERTION
int number_new_nodes;
strings_node new_nodes[MAX_DIFFERENT_NEW_NODES];
#endif
strings_hash_table strings_ht[STRINGS_HT_NUMBER];
static inline int hashtag_to_bucket(const char *hashtag)
{
int i = 0;
unsigned long long int bucket = 0ULL;
while (hashtag[i] != '\0')
{
bucket = (bucket + (PRIME_NUMBER * ((unsigned long long int) hashtag[i]))) % STRINGS_HT_BUCKETS;
i++;
}
return (int) bucket;
}
static inline int insert_hashtag_into_requests(char *hashtag, size_t hashtag_length)
{
int ret;
int prob_k_i;
ret = 1;
if (number_requested_hashtags < MAX_DIFFERENT_REQUESTED_HASHTAGS)
memcpy((void *) requested_hashtags[number_requested_hashtags], (const void *) hashtag, hashtag_length);
else
{
prob_k_i = (rand() % number_requested_hashtags);
if (prob_k_i < MAX_DIFFERENT_REQUESTED_HASHTAGS)
memcpy((void *) requested_hashtags[prob_k_i], (const void *) hashtag, hashtag_length);
else
ret = 0;
}
number_requested_hashtags++;
return ret;
}
#ifdef NEW_NODES_INSERTION
static inline int insert_node_into_new_nodes(strings_node *node)
{
int ret;
int prob_k_i;
strings_node *nd;
ret = 1;
if (number_new_nodes < MAX_DIFFERENT_NEW_NODES)
{
memcpy((void *) &new_nodes[number_new_nodes], (const void *) node, sizeof(strings_node));
new_nodes[number_new_nodes].next = NULL;
new_nodes[number_new_nodes].prev = NULL;
}
else
{
prob_k_i = (rand() % number_new_nodes);
if (prob_k_i < MAX_DIFFERENT_NEW_NODES)
{
memcpy((void *) &new_nodes[prob_k_i], (const void *) node, sizeof(strings_node));
new_nodes[number_new_nodes].next = NULL;
new_nodes[number_new_nodes].prev = NULL;
}
else
ret = 0;
}
number_new_nodes++;
return ret;
}
#endif
int insert_node_into_hashtable(strings_node *node)
{
size_t hashtag_length;
size_t hashtags_length;
strings_node *nd;
strings_node **lst_addr;
strings_hash_table *sht;
char *hashtag;
char hashtags[(hashtags_length = strlen((const char *) node->hashtags) + 1)];
memcpy((void *) hashtags, (const void *) node->hashtags, hashtags_length);
sht = &(strings_ht[(unsigned int) (rand() % STRINGS_HT_NUMBER)]);
hashtag = strtok(hashtags, " ");
while (hashtag != NULL)
{
hashtag_length = strlen((const char *) hashtag) + 1;
lst_addr = &(sht->list[hashtag_to_bucket((const char *) hashtag)]);
if ((nd = (strings_node *) malloc(sizeof(strings_node))) == NULL)
return -1;
memcpy((void *) nd, (const void *) node, sizeof(strings_node));
nd->prev = NULL;
if ((*lst_addr) != NULL)
{
nd->next = (struct __strings_node__ *) (*lst_addr);
nd->next->prev = (struct __strings_node__ *) nd;
}
else
{
nd->next = NULL;
}
(*lst_addr) = nd;
#ifdef NEW_NODES_INSERTION
if (insert_hashtag_into_requests(hashtag, hashtag_length) == 0)
{
insert_node_into_new_nodes(node);
}
#else
insert_hashtag_into_requests(hashtag, hashtag_length);
#endif
hashtag = strtok(NULL, " ");
}
return 0;
}
int strings_hash_table_init(void)
{
FILE *file;
int i, j;
int scan_items;
int inserted_nodes;
int hashtags_text_turn;
char *line, *header, *value;
strings_node *node;
#if COMPLETED_REQUESTS_MONITORING
completed_requests = 0ULL;
#endif
number_requested_hashtags = 0;
for (i=0; i<MAX_DIFFERENT_REQUESTED_HASHTAGS; i++)
{
requested_hashtags[i][0] = '\0';
}
#ifdef NEW_NODES_INSERTION
number_new_nodes = 0;
for (i=0; i<MAX_DIFFERENT_NEW_NODES; i++)
{
memset((void *) &new_nodes[i], 0, sizeof(strings_node));
}
#endif
for (i=0; i<STRINGS_HT_NUMBER; i++)
{
for (j=0; j<STRINGS_HT_BUCKETS; j++)
{
strings_ht[i].list[j] = NULL;
#ifdef NEW_NODES_INSERTION
omp_init_lock(&strings_ht[i].lock[j]);
#endif
}
}
if ((file = fopen("TextHashtags.txt", "r")) == NULL)
return -1;
node = NULL;
inserted_nodes = 0;
hashtags_text_turn = 1;
while ((scan_items = fscanf(file, "%m[^\n]\n", &line)) != EOF)
{
header = strtok(line, ":");
value = strtok(NULL, "\0");
if (header == NULL || value == NULL)
{
free((void *) line);
continue;
}
if (hashtags_text_turn)
{
if (strcmp("HASHTAGS", (const char *) header) != 0)
{
free((void *) line);
continue;
}
if (node != NULL || (node = (strings_node *) malloc(sizeof(strings_node))) == NULL)
goto strings_hash_table_init_error;
strncpy(node->hashtags, (const char *) value, (size_t) HASHTAG_MAX_LENGTH-1);
node->hashtags[HASHTAG_MAX_LENGTH-1] = '\0';
hashtags_text_turn = 0;
}
else
{
if (strcmp("TEXT", (const char *) header) != 0)
{
free((void *) line);
continue;
}
if (node == NULL)
goto strings_hash_table_init_error;
strncpy(node->text, (const char *) value, (size_t) TEXT_MAX_LENGTH-1);
node->text[TEXT_MAX_LENGTH-1] = '\0';
if (insert_node_into_hashtable(node))
goto strings_hash_table_init_error;
if ((++inserted_nodes % 1000) == 0)
{
printf("\rInserted Nodes: %u", inserted_nodes);
fflush(stdout);
}
free((void *) node);
node = NULL;
hashtags_text_turn = 1;
}
free((void *) line);
}
if ((j = number_requested_hashtags) < MAX_DIFFERENT_REQUESTED_HASHTAGS)
for (i=j; i<MAX_DIFFERENT_REQUESTED_HASHTAGS; i++)
insert_hashtag_into_requests(requested_hashtags[(i-j)], strlen(requested_hashtags[(i-j)]));
number_requested_hashtags = 0;
#ifdef NEW_NODES_INSERTION
if ((j = number_new_nodes) < MAX_DIFFERENT_NEW_NODES)
for (i=j; i<MAX_DIFFERENT_NEW_NODES; i++)
insert_node_into_new_nodes(&new_nodes[(i-j)]);
number_new_nodes = 0;
#endif
if (fclose(file))
return -1;
return 0;
strings_hash_table_init_error:
if (fclose(file))
return -1;
return -1;
}
void strings_hash_table_fini(void)
{
int removed_nodes;
unsigned int i;
unsigned int j;
strings_node *nd;
strings_node *nd_aux;
removed_nodes = 0;
for (i=0; i<STRINGS_HT_NUMBER; i++)
{
for (j=0; j<STRINGS_HT_BUCKETS; j++)
{
nd = strings_ht[i].list[j];
while (nd != NULL)
{
nd_aux = (strings_node *) nd->next;
free((void *) nd);
nd = nd_aux;
if ((++removed_nodes % 1000) == 0)
{
printf("\rRemoved Nodes: %u", removed_nodes);
fflush(stdout);
}
}
strings_ht[i].list[j] = NULL;
#ifdef NEW_NODES_INSERTION
omp_destroy_lock(&strings_ht[i].lock[j]);
#endif
}
}
}
static inline int is_hashtag_contained(char *hashtags, char *hashtag, int hashtag_length)
{
int i;
int j;
int contained;
char *hashtag_aux;
i = 0;
while (1)
{
if (hashtags[i] == '\0')
{
return 0;
}
else if (hashtags[i] == ' ')
{
i++;
continue;
}
hashtag_aux = &(hashtags[i]);
i++;
j = 1;
while (hashtags[i] != '\0' && hashtags[i] != ' ')
{
i++;
j++;
}
if (j != hashtag_length)
{
if (hashtags[i] == ' ')
i++;
}
else
{
contained = 1;
for (j=0; j<hashtag_length; j++)
{
if (hashtag_aux[j] != hashtag[j])
{
contained = 0;
break;
}
}
if (contained)
return 1;
if (hashtags[i] == ' ')
i++;
}
}
}
char ** query_hashtable(char *hashtag, int i)
{
int j;
int len;
int bucket;
int num_results;
char **hashtags;
#if COMPLETED_REQUESTS_MONITORING
unsigned long long int local;
#endif
strings_node *nd;
strings_hash_table *sht;
if ((hashtags = (char **) malloc(MAX_RESULTS_PER_HASHTAG * sizeof(char *))) == NULL)
return NULL;
for (j=0; j<MAX_RESULTS_PER_HASHTAG; j++)
hashtags[j] = NULL;
len = 0;
while (hashtag[len] != '\0') len++;
bucket = hashtag_to_bucket((const char *) hashtag);
sht = &(strings_ht[i]);
#ifdef NEW_NODES_INSERTION
omp_set_lock(&sht->lock[bucket]);
#endif
nd = sht->list[bucket];
num_results = 0;
while (nd != NULL)
{
if (num_results == MAX_RESULTS_PER_HASHTAG)
break;
if (is_hashtag_contained(nd->hashtags, hashtag, len))
{
hashtags[num_results] = nd->text;
num_results++;
}
nd = (strings_node *) nd->next;
}
#ifdef NEW_NODES_INSERTION
omp_unset_lock(&sht->lock[bucket]);
#endif
#if COMPLETED_REQUESTS_MONITORING
local = __atomic_add_fetch (&completed_requests, 1, __ATOMIC_RELAXED);
if ((local % 1000) == 0)
{
printf("\rCompleted: %llu", local);
fflush(stdout);
}
#endif
return hashtags;
}
char *** query_hashtables(char *hashtag, int priority)
{
int i;
char ***texts;
if ((texts = (char ***) malloc(STRINGS_HT_NUMBER * sizeof(char **))) == NULL)
return NULL;
for (i=0; i<STRINGS_HT_NUMBER; i++)
#if MANUAL_CUT_OFF
texts[i] = query_hashtable(hashtag, i);
#else
{
#pragma omp task TIED_QUERY_HASHTABLE priority(priority)
texts[i] = query_hashtable(hashtag, i);
}
#pragma omp taskwait
#endif
return texts;
}
#ifdef NEW_NODES_INSERTION
void insert_new_node_into_list(strings_hash_table *sht, int bucket, strings_node *node)
{
size_t s;
char *src;
char *dst;
#if COMPLETED_REQUESTS_MONITORING
unsigned long long int local;
#endif
strings_node *nd;
strings_node **lst_addr;
if ((nd = (strings_node *) malloc(sizeof(strings_node))) == NULL)
return;
src = (char *) node;
dst = (char *) nd;
for (s=0; s<sizeof(strings_node); s++)
dst[s] = src[s];
nd->prev = NULL;
omp_set_lock(&sht->lock[bucket]);
lst_addr = &(sht->list[bucket]);
if ((*lst_addr) == NULL)
nd->next = NULL;
else
{
nd->next = (struct __strings_node__ *) (*lst_addr);
nd->next->prev = (struct __strings_node__ *) nd;
}
(*lst_addr) = nd;
omp_unset_lock(&sht->lock[bucket]);
#if COMPLETED_REQUESTS_MONITORING
local = __atomic_add_fetch (&completed_requests, 1, __ATOMIC_RELAXED);
if ((local % 1000) == 0)
{
printf("\rCompleted: %llu", local);
fflush(stdout);
}
#endif
}
void insert_new_node_into_hashtable(strings_node *node)
{
int i;
size_t s;
size_t hashtags_length;
int bucket;
char *hashtag;
strings_hash_table *sht;
hashtags_length = 0;
while (node->hashtags[hashtags_length] != '\0')
hashtags_length++;
char hashtags[(hashtags_length + 1)];
for (s=0; s<(hashtags_length + 1); s++)
hashtags[s] = node->hashtags[s];
sht = &(strings_ht[(unsigned int) (rand() % STRINGS_HT_NUMBER)]);
i = 0;
while (1)
{
if (hashtags[i] == '\0')
{
break;
}
else if (hashtags[i] == ' ')
{
i++;
continue;
}
hashtag = &hashtags[i];
while (hashtags[i] != '\0' && hashtags[i] != ' ')
i++;
if (hashtags[i] == ' ')
{
hashtags[i] = '\0';
i++;
}
bucket = hashtag_to_bucket((const char *) hashtag);
#if MANUAL_CUT_OFF
insert_new_node_into_list(sht, bucket, node);
#else
#pragma omp task TIED_INSERT_HASHTABLE
insert_new_node_into_list(sht, bucket, node);
#endif
}
}
#endif
void results_finalization(char ****results)
{
int i;
int j;
for (i=0; i<REQUESTS_NUMBER; i++)
{
if (results[i] != NULL)
{
for (j=0; j<STRINGS_HT_NUMBER; j++)
{
if (results[i][j] != NULL)
free((void *) results[i][j]);
}
free((void *) results[i]);
}
}
}
static inline int get_priority(void)
{
double random_0_to_1 = (double) rand() / (double) RAND_MAX;
if (random_0_to_1 <= PRIORITY_1_PROBABILITY)
{
return 1;
}
else if (random_0_to_1 <= (PRIORITY_1_PROBABILITY + PRIORITY_2_PROBABILITY))
{
return 3;
}
else if (random_0_to_1 <= (PRIORITY_1_PROBABILITY + PRIORITY_2_PROBABILITY + PRIORITY_3_PROBABILITY))
{
return 5;
}
return -1;
}
#ifdef POISSON_ARRIVAL_TIME
static inline useconds_t poisson_process_next_usec_time(void)
{
return (useconds_t) (-1000000.0 * (log(((double) rand() / (double) RAND_MAX)) / REQUEST_ARRIVALS_RATE));
}
#endif
int main(int argc, char *argv[])
{
int i;
int ret;
int priority;
char ****results;
srand((unsigned int) PSEUDO_RANDOM_SEED);
omp_set_text_section_addresses(&_start, &_fini);
printf("\nStarting Hash-Tables Initialization\n");
if ((ret = strings_hash_table_init()))
goto parallel_hash_tables_exit;
printf("\nInitialization Completed\n");
if ((results = (char ****) malloc(REQUESTS_NUMBER * sizeof(char ***))) == NULL)
goto parallel_hash_tables_exit;
for (i=0; i<REQUESTS_NUMBER; i++)
results[i] = NULL;
#pragma omp parallel
{
#pragma omp single private(i, priority)
{
for (i=0; i<REQUESTS_NUMBER; i++)
{
#ifdef NEW_NODES_INSERTION
if (((double) rand() / (double) RAND_MAX) < NEW_NODES_INSERION_PROBABILITY)
{
#pragma omp task TIED_INSERT_REQUEST
insert_new_node_into_hashtable(&new_nodes[number_new_nodes]);
number_new_nodes = ((number_new_nodes + 1) % MAX_DIFFERENT_NEW_NODES);
}
else
#endif
{
if ((priority = get_priority()) == -1)
{
printf("\nPriority Generation Failure at step %d\n", i);
break;
}
#pragma omp task TIED_QUERY_REQUEST priority(priority)
results[number_requested_hashtags] = query_hashtables(requested_hashtags[number_requested_hashtags], priority+1);
number_requested_hashtags = ((number_requested_hashtags + 1) % MAX_DIFFERENT_REQUESTED_HASHTAGS);
}
#ifdef POISSON_ARRIVAL_TIME
if (i < (REQUESTS_NUMBER - 1) && usleep(poisson_process_next_usec_time()))
{
printf("\nPoisson Process Failure at step %d\n", i);
break;
}
#endif
}
}
}
results_finalization(results);
free((void *) results);
parallel_hash_tables_exit:
printf("\nStarting Hash-Tables Finalization\n");
strings_hash_table_fini();
printf("\nFinalization Completed\n\n");
return ret;
}