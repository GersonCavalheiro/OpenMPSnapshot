#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "payload.h"
static int 
count_zeroes(const unsigned char *hash, const int len)
{
int cnt = 0;
for (int i = 0; i < len; i++) {
cnt += hash[i] == 0;
}
return cnt;
}
static int 
run_omp(const char *str, const int target_count, const int n_threads)
{
omp_set_num_threads(n_threads);
int ret = 0;
#pragma omp parallel reduction(+ : ret)
{
struct payload_t *p = payload_gen(str);
assert(p);
unsigned char hash[32];
#pragma omp for schedule(dynamic)
for (long i = 0; i < MAX_MAGIC; i++) {
payload_set_magic(p, i);
payload_checksum(p, hash);
int cnt = count_zeroes(hash, sizeof(hash));
ret += cnt == target_count;
}
payload_free(p);
}
return ret;
}
struct 
arguments
{
pthread_t* threads;
pthread_mutex_t m1;
pthread_mutex_t m2;
int ret;
long curarg;
int break_all;
const char* str;
int target_count;
};
void * 
start_routine(void* pl)
{	
struct arguments* pool = (struct arguments*)pl;
struct payload_t *p = payload_gen(pool->str);
if (p == NULL)
{
fprintf(stderr,"ERROR at line %d\n\t\tat function %s\n\t\
Description: %s",__LINE__, __FUNCTION__,"payload == NULL");
exit(__LINE__);
}
unsigned char hash[32];
while(1)
{
pthread_mutex_lock(&(pool->m1));
if(pool->break_all)
{
pthread_mutex_unlock(&(pool->m1));
payload_free(p);
pthread_exit(0);
}
pool->curarg++;
if (pool->curarg >= MAX_MAGIC)
{
pool->break_all = 1;
pthread_mutex_unlock(&(pool->m1));
continue;
}
payload_set_magic(p, pool->curarg);
pthread_mutex_unlock(&(pool->m1));
payload_checksum(p, hash);
int cnt = count_zeroes(hash, sizeof(hash));
pthread_mutex_lock(&(pool->m2));
pool->ret+= cnt == pool->target_count;
pthread_mutex_unlock(&(pool->m2));
}
}
static int 
run_pthreads(const char *str, const int target_count, const int n_threads)
{
struct arguments pool;
pool.curarg=0;
pool.break_all = 0;
pool.threads = (pthread_t *)malloc(n_threads * sizeof(pthread_t));
if(pool.threads  == NULL)
return -1;
pthread_mutex_init(&(pool.m1),NULL);
pthread_mutex_init(&(pool.m2), NULL);
pool.str = str;
pool.target_count = target_count;
pool.ret = 0;
for(int i = 0; i < n_threads; i++)
{
if(pthread_create(&(pool.threads[i]),NULL, start_routine, &pool))
return -1;
}
for(int i = 0; i < n_threads; i++)
{
if(pthread_join(pool.threads[i], NULL) != 0)
return -1;
}
pthread_mutex_destroy(&pool.m1);
pthread_mutex_destroy(&pool.m2);
free(pool.threads);
return pool.ret;
}
typedef int (*cb)(const char *str, const int target_count, const int n_threads);
struct 
result_t 
{
double elapsed;
int cnt;
};
static struct result_t 
timer(const cb f, const char *str, const int target_count, const int n_threads)
{
struct result_t res;
double start = omp_get_wtime();
res.cnt = f(str, target_count, n_threads);
res.elapsed = omp_get_wtime() - start;
return res;
}
static int 
check(const char *str, const int target_count, const int n_threads) {
struct result_t r1 = timer(run_omp, str, target_count, n_threads);
printf("OpenMP: cnt = %d, elapsed = %lfs\n", r1.cnt, r1.elapsed);
struct result_t r2 = timer(run_pthreads, str, target_count, n_threads);
printf("pthreads: cnt = %d, elapsed = %lfs\n", r2.cnt, r2.elapsed);
if (r1.cnt != r2.cnt){
printf("Unexpected count: got %d, want %d\n", r2.cnt, r1.cnt);
return -1;
}
if (0.9 * r2.elapsed > r1.elapsed){
printf("pthreads version is %lf times slower than OpenMP one\n", r2.elapsed / r1.elapsed);
return -2;
}
return 0;
}
int 
main(int argc, char **argv)
{
char *data = argv[1];
int target_count = atoi(argv[2]);
int n_threads = atoi(argv[3]);
return check(data, target_count, n_threads);
}
