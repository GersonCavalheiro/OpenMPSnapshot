#include "async_struct.h"
#include "task_log.h"
#include "prof.h"
#include <stdlib.h>
#include <stdio.h>
void async_log(async_t *sync, unsigned int size, const char *id) 
{
sync->logc = 0;
sync->logs = size;
sync->log = malloc(size * sizeof(log_t));	
char fname[32];
snprintf(fname, 32, "%s_%i.log", id, sync->id);
sync->logf = fopen(fname, "w");
}
async_t * async_init(async_t *sync, int c, int n, int b, int initc, int log)
{
int init = sync[0].id == -1;
int wait = 0;
int i;
for ( i=0; i<c; ++i ) {
int create  = sync[i].create;  
int consume = sync[i].consume;  
if ( !init && create != consume && !wait ) {
#pragma omp taskwait	
wait = 1;
} 
int bc = (n + b - 1) / b;
sync[i].create = initc;
sync[i].ccnt = 0;
sync[i].pcnt = 0;
sync[i].pcompl = bc;
sync[i].consume = initc;
sync[i].wait = 0;
sync[i].flags = 0;
sync[i].ready = initc;
sync[i].dot_control = initc;
sync[i].log = NULL;
sync[i].logf = NULL;
sync[i].logs = 0;
sync[i].prof.s = 0;
if ( init ) {
pthread_mutex_init(&sync[i].mutex, NULL);
pthread_cond_init(&sync[i].cond, NULL);
sync[i].id = i;
}
}
return sync;
}
void async_profile(async_t *sync, float prof)
{
prof_malloc(&(sync->prof), prof, sync->pcompl, 2);
}
void async_fini(async_t *sync, int c) {
int i;
for ( i=0; i<c; ++i ) {
pthread_mutex_destroy(&sync[i].mutex);
pthread_cond_destroy(&sync[i].cond);
if ( sync[i].log ) {
printf("warn: writing log %i\n", i);
log_flush(sync[i].logf, sync[i].log, sync[i].logc);
fclose(sync[i].logf);
free(sync[i].log);
}
prof_free(&sync[i].prof);
}
}
