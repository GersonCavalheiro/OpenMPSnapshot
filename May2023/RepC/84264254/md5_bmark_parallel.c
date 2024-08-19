#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include "md5.h"
#include "md5_bmark.h"
typedef struct timeval timer;
int nt; 
#define TIME(x) gettimeofday(&x, NULL)
int initialize(md5bench_t* args);
int finalize(md5bench_t* args);
void run(md5bench_t* args);
void process(uint8_t* in, uint8_t* out, int bufsize);
void listInputs();
long timediff(timer* starttime, timer* finishtime);
static data_t datasets[] = {
{64, 512, 0},
{64, 1024, 0},
{64, 2048, 0},
{64, 4096, 0},
{128, 1024*512, 1},
{128, 1024*1024, 1},
{128, 1024*2048, 1},
{128, 1024*4096, 1},
};
int initialize(md5bench_t* args) {
int index = args->input_set;
if(index < 0 || index >= sizeof(datasets)/sizeof(datasets[0])) {
fprintf(stderr, "Invalid input set specified! Clamping to set 0\n");
index = 0;
}
args->numinputs = datasets[index].numbufs;
args->size = datasets[index].bufsize;
args->inputs = (uint8_t**)calloc(args->numinputs, sizeof(uint8_t*));
args->out = (uint8_t*)calloc(args->numinputs, DIGEST_SIZE);
if(args->inputs == NULL || args->out == NULL) {
fprintf(stderr, "Memory Allocation Error\n");
return -1;
}
srand(datasets[index].rseed);
for(int i = 0; i < args->numinputs; i++) {
args->inputs[i] = (uint8_t*)malloc(sizeof(uint8_t)*datasets[index].bufsize);
uint8_t *p = args->inputs[i];
if(p == NULL) {
fprintf(stderr, "Memory Allocation Error\n");
return -1;
}
for(int j = 0; j < datasets[index].bufsize; j++)
*(p + j) = rand() % 255;
}
return 0;
}
void process(uint8_t* in, uint8_t* out, int bufsize) {
MD5_CTX context;
uint8_t digest[16];
MD5_Init(&context);
MD5_Update(&context, in, bufsize);
MD5_Final(digest, &context);
memcpy(out, digest, DIGEST_SIZE);
}
void run(md5bench_t* args) {
for(int i = 0; i < args->iterations; i++) {
int buffers_to_process = args->numinputs;
int next = 0;
uint8_t** in = args->inputs;
uint8_t* out = args->out;
#pragma omp parallel num_threads(nt)
{		
#pragma omp for
for (next = 0; next < buffers_to_process; next++) {
#pragma omp task
process(in[next], out+next*DIGEST_SIZE, args->size);
}
}
}
}
int finalize(md5bench_t* args) {
char buffer[64];
for(int i = 0; i < args->numinputs; i++) {
#ifdef DEBUG
sprintf(buffer, "Buffer %d has checksum ", i);
fwrite(buffer, sizeof(char), strlen(buffer)+1, stdout);
#endif
for(int j = 0; j < DIGEST_SIZE*2; j+=2) {
sprintf(buffer+j,   "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf);
sprintf(buffer+j+1, "%x", args->out[DIGEST_SIZE*i+j/2] & 0xf0);
}
buffer[32] = '\0';
#ifdef DEBUG            
fwrite(buffer, sizeof(char), 32, stdout);
fputc('\n', stdout);
#else
printf("%s ", buffer);
#endif
}
#ifndef DEBUG
printf("\n");
#endif
if(args->inputs) {
for(int i = 0; i < args->numinputs; i++) {
if(args->inputs[i])
free(args->inputs[i]);
}
free(args->inputs);
}
if(args->out)
free(args->out);
return 0;
}
long timediff(timer* starttime, timer* finishtime)
{
long msec;
msec=(finishtime->tv_sec-starttime->tv_sec)*1000;
msec+=(finishtime->tv_usec-starttime->tv_usec)/1000;
return msec;
}
int main(int argc, char** argv) {
timer b_start, b_end;
md5bench_t args;
scanf("%d", &nt);
scanf("%d", &args.input_set);
scanf("%d", &args.iterations);
args.outflag = 1;
if(initialize(&args)) {
fprintf(stderr, "Initialization Error\n");
exit(EXIT_FAILURE);
}
TIME(b_start);
run(&args);
TIME(b_end);
if(finalize(&args)) {
fprintf(stderr, "Finalization Error\n");
exit(EXIT_FAILURE);
}
double b_time = (double)timediff(&b_start, &b_end)/1000;
printf("%.3f\n", b_time);
return 0;
}
