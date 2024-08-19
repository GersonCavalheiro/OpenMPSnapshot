#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +     \
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
#if defined(VERBOSE)
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif
#define MAX( x, y ) ((x) > (y) ? (x) : (y))
#define N_def             100
#define N_roots_def       0.02
#define min_children_def  1
#define max_children_def  4
#define min_datasize_def  1
#define max_datasize_def  100
#define workload_def      1000
char filename_def[] = "dag.dot";
typedef unsigned int uint;
typedef struct
{
int    n_ancestors;    
int    u_ancestors;    
int    n_children;     
uint   counter;        
uint  *children;       
uint   datasize;
char  *data;           
double result;         
} node_t;
typedef struct
{
uint   N;              
uint   N_roots;        
int    min_children;   
int    max_children;   
uint   min_datasize;   
uint   max_datasize;   
uint   workload;       
uint  *children_lists; 
char  *filename;       
} dag_t;
char **storages;
uint  *storages_size;
uint  *storages_top;
typedef short unsigned int seeds_t[3];
typedef short unsigned int *seeds_pt;
seeds_t *seeds;          
node_t * generate_dag(dag_t *);
double   heavy_work( int N );
void update_node( node_t *nodes, node_t *node, uint *check, uint workload )
{
uint work = workload / (node->n_children+1);
#if !defined(REPRODUCIBLE)  
work = workload / 100 + nrand48((seeds_pt)&seeds[omp_get_thread_num()]) % work;
#endif
for ( int j = 0; j < node->n_children; j++ )
{
int u_ancestors;
uint idx = node->children[j];      
#pragma omp atomic update
nodes[idx].counter += work;
#pragma omp atomic read
u_ancestors = nodes[idx].u_ancestors;
#pragma omp atomic update
nodes[idx].u_ancestors++;
if ( nodes[idx].n_ancestors - u_ancestors == 1 )
#pragma omp task
update_node( nodes, &nodes[idx], check, workload );
}
for( int i = 0; i < node->datasize; i++ )
node->data[i]++;
node->result = heavy_work( node->counter);
#pragma omp atomic update
(*check)++;
node-> u_ancestors = 0;
return;
}
uint process_dag( node_t *nodes, dag_t *dag )
{
uint done = 0;
int  nthreads;
#pragma omp parallel
#pragma omp single
nthreads = omp_get_num_threads();
PRINTF("- using %d threads\n", nthreads);
seeds = (seeds_t*)malloc( nthreads * sizeof(seeds_t) );
#pragma omp parallel shared(seeds, done)
{
int me = omp_get_thread_num();
short int myseeds[3] = {me*123, me*123+1, me*123+2};
seed48((seeds_pt)&seeds[me]);    
#pragma omp single
{
for ( int j = 0; j < dag->N_roots; j++ )
{
uint work = dag->workload / (nodes[j].n_children+1);
#if !defined(REPRODUCIBLE)
work = dag->workload / 100 + nrand48((seeds_pt)&seeds[omp_get_thread_num()]) % work;
#endif
nodes[j].counter = work;
#pragma omp task
update_node( nodes, &nodes[j], &done, dag->workload );
}
#pragma omp taskwait
PRINTF("- thread %d has generated the first %d tasks for the root nodes;\n"
"  tasks have been completed, now it is joining the pool\n",
me, dag->N_roots );
}
}
PRINTF("- %u nodes have been done\n", done );
uint undone = 0;
for ( int n = 0; n < dag->N; n++ )
undone += (nodes[n].counter > 0 && nodes[n].u_ancestors > 0 );
if ( undone > 0 )
PRINTF("* oops, no something went wrong: %u nodes are undone\n", undone);
return undone;
}
int main ( int argc, char **argv )
{
dag_t dag = { N_def, (uint)(N_def*N_roots_def+1), min_children_def, max_children_def, min_datasize_def, max_datasize_def, workload_def, NULL, NULL};
int write_dag = 0;
int a = 0;
if ( argc > ++a ) {
if ( (write_dag = atoi(*(argv+a)) ) )
dag.filename = filename_def;
if ( argc > ++a ) {
dag.N = (uint)(atoi(*(argv+a)));
if ( argc > ++a ) {
dag.N_roots = (uint)(dag.N * (float)(atof(*(argv+a)))+1);
if ( argc >= ++a ) {
dag.min_children = (uint)(atoi(*(argv+a)));
dag.max_children = (uint)(atoi(*(argv+ (++a))));
if ( argc >= ++a ) {
dag.min_datasize = (uint)(atoi(*(argv+a)));
dag.max_datasize = (uint)(atoi(*(argv+ (++a))));	    
if ( argc >= ++a )
dag.workload = (uint)(atoi(*(argv+a)));
} } } } }
node_t *nodes = generate_dag( &dag );
if ( write_dag )
return 0;
struct  timespec ts;
double wtstart = CPU_TIME;
process_dag( nodes, &dag );
double wtend = CPU_TIME;
printf("%g sec elapsed\n", wtend - wtstart);
free( dag.children_lists );
free( nodes );
return 0;
}
node_t * generate_dag(dag_t *p)
{
FILE   *file      = NULL;
node_t *nodes;
uint    cidx      = 0;
uint    avg_n_children = 0;
int     nthreads;
PRINTF("generating the DAG with %u nodes..\n", p->N );
int     sidx = 0;
long long unsigned int total_datasize = 0;
uint    min_datasize = p->max_datasize;
uint    max_datasize = p->min_datasize;
#pragma omp parallel
{   
#pragma omp single
nthreads  = omp_get_num_threads();
}
#if defined(DISTRIBUTE_MEMORY)
nodes     = (node_t*)malloc( p->N * sizeof(node_t) );
p->children_lists = (uint*)malloc( p->N * p->max_children* sizeof(uint) );
uint initial_datasize = (1024 * 0.6*(p->min_datasize + p->max_datasize)/2.0 * p->N) / nthreads;
storages      = (char**)malloc( nthreads * sizeof(char*) );
storages_size = (uint*)malloc( nthreads * sizeof(uint) );
storages_top  = (uint*)malloc( nthreads * sizeof(uint) );
for ( int i = 0; i < nthreads; i++ ) {
storages_size[i] = initial_datasize; storages_top[i] = 0;} 
#pragma omp parallel
{   
int me    = omp_get_thread_num();
int rem   = nthreads % nthreads;
int bunch = p->N / nthreads;
int start   = (bunch + (me < rem))*me;
int mybunch = bunch + (me < rem);
memset( &nodes[start], 0, sizeof(node_t)*mybunch);
memset( p->children_lists, 0, sizeof(uint)*mybunch);
storages[me] = (char*)calloc( initial_datasize,  sizeof(char) );
}
#else
nodes             = (node_t*)calloc( p->N, sizeof(node_t) );
p->children_lists = (uint*)calloc( p->N * p->max_children, sizeof(uint) );
uint  initial_datasize = 1024 * 0.6*(p->min_datasize + p->max_datasize)/2.0 * p->N;
uint  strg_size        = initial_datasize;
uint  strg_top         = 0;
char *strg             = (char*)calloc( initial_datasize, sizeof(char) );  
storages      = &strg;  
storages_size = &strg_size;
storages_top  = &strg_top;
#endif
if ( p->filename != NULL )
{
file = fopen( p->filename, "w");
if ( file == NULL ) {
printf("\terror creating file %s\n", p->filename); return NULL; }
fprintf( file, "digraph {\n");
}
srand48(time(NULL));
for ( uint n = 0; n < p->N-2; n++ )
{
int off   = MAX(p->N_roots, n) + 1;
int tail  = p->N - off;
if ( tail < p->min_children )
break;
if ( (n >= p->N_roots) && (nodes[n].n_ancestors == 0) ) {
nodes[n-1].children[nodes[n-1].n_children++] = n;
cidx++;
avg_n_children++;
nodes[n].n_ancestors = 1;
}
nodes[n].n_children = p->min_children + ( lrand48() % (p->max_children - p->min_children) );
nodes[n].datasize = p->min_datasize + ( lrand48() % (p->max_datasize - p->min_datasize) );
total_datasize += nodes[n].datasize;
min_datasize = (min_datasize > nodes[n].datasize ? nodes[n].datasize : min_datasize );
max_datasize = (max_datasize < nodes[n].datasize ? nodes[n].datasize : max_datasize );
{
#if defined(DISTRIBUTED_MEMORY)
int found = 0;
int guard_sidx = sidx-1;
while ( !found && (current_sidx < sidx) ) {
found =  (storages_top[sidx] + nodes[n].datasize <= storages_size[sidx]);
if( !found ) {
char *new_storage = (char*)realloc( storages[sidx], storages_size[sidx]+nodes[n].datasize );
if ( new_storage != NULL ) { found = 1; storages[sidx] = new_storage; }
else sidx = (++sidx % nthreads); } }
if ( current_sidx == sidx )
{
printf(">>>>>>>>>> not enough memory found on any thread\n" );
exit(1);
}
char *start = &storages[sidx][storages_top[sidx]];
char *stop  = start + nodes[n].datasize;
#else
if( storages_top[0] + nodes[n].datasize <= storages_size[0] ) {
char *new_storage = (char*)realloc( storages[0], storages_size[0]+nodes[n].datasize );
if ( new_storage != NULL ) storages[0] = new_storage;
else { printf(">>>>>>>>>> not enough memory found on any thread\n" ); exit(1); } }
char *start = storages[0] + storages_top[0];
char *stop  = start + nodes[n].datasize;
#endif
nodes[n].data = start;
for( ; start < stop; start++ )
*start = sidx;
sidx = (++sidx % nthreads);
}
if ( nodes[n].n_children > tail )
nodes[n].n_children = tail;
avg_n_children += nodes[n].n_children;
nodes[n].children = p->children_lists + cidx;
cidx += nodes[n].n_children;
for ( int j = 0; j < nodes[n].n_children; j++ )
{
int idx   = off + lrand48() % tail;
nodes[idx].n_ancestors++;
nodes[n].children[j] = idx;
if ( file != NULL )
fprintf( file, "  %u -> %u;\n", n, idx );
}
}
PRINTF("\tthe avg # of children is %3.1f\n", (double)avg_n_children / p->N );
PRINTF("\t%Lu bytes have been allocated (avg %f, min %u, max %u)\n",
total_datasize, (double)total_datasize/p->N, min_datasize, max_datasize );
#if !defined(DISTRIBUTE_MEMORY)
if ( cidx < p->N * p->max_children )
{
uint size = p->N*p->max_children;
PRINTF("\treducing children-lists size from %u to %u entries (%3.2f%%)\n",
size, cidx, ((double)(size-cidx))*100 / size);
uint *old_base = p->children_lists;
uint *new_base = (uint*)realloc(p->children_lists, cidx*sizeof(uint));
if ( new_base != NULL )
{
p->children_lists = new_base;
for ( int n = 0; n < p->N; n++ )
{
size_t off = nodes[n].children - old_base;
nodes[n].children = p->children_lists + off;
}
}
}
#endif
{
uint zerocount = 0;
for ( int n = p->N_roots; n < p->N; n++ )
zerocount+= ( nodes[n].n_ancestors == 0 );
if ( zerocount > 0 )
PRINTF("\t%u nodes with no ancestors found\n", zerocount);
uint before = 0;
for ( int n = p->N_roots; n < p->N; n++ )
for ( int j = 0; j < nodes[n].n_children; j++ )
if ( nodes[n].children[j] <= n )
before++;
if ( before > 0 )
PRINTF("\t%u nodes with uncorrect children indexes\n", before);
}
if ( file != NULL )
{
fprintf( file, "}\n");
fclose(file);
}
return nodes;
}
double heavy_work( int N )
{
double guess = 3.141572 / 5 * N;
guess = ( guess > 200 ? 111 : guess);
for( int i = 0; i < N; i++ )
{
guess = exp( guess );
guess = sin( guess );
}
return guess;
}
