#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <omp.h>
#include "radixsort.h"
#include "edgeList.h"
#include "vertex.h"
#include "myMalloc.h"
#include "graphConfig.h"
#include "timer.h"
void radixSortCountSortEdgesBySource (struct EdgeList **sorted_edges_array, struct EdgeList **edgeList, uint32_t radix, uint32_t buckets, uint32_t *buckets_count)
{
struct EdgeList *temp_edges_array = NULL;
uint32_t num_edges = (*edgeList)->num_edges;
uint32_t t = 0;
uint32_t o = 0;
uint32_t u = 0;
uint32_t i = 0;
uint32_t j = 0;
uint32_t P = 1;  
uint32_t t_id = 0;
uint32_t offset_start = 0;
uint32_t offset_end = 0;
uint32_t base = 0;
#pragma omp parallel default(none) shared(sorted_edges_array,edgeList,radix,buckets,buckets_count,num_edges) firstprivate(t_id, P, offset_end,offset_start,base,i,j,t,u,o)
{
P = omp_get_num_threads();
t_id = omp_get_thread_num();
offset_start = t_id * (num_edges / P);
if(t_id == (P - 1))
{
offset_end = offset_start + (num_edges / P) + (num_edges % P) ;
}
else
{
offset_end = offset_start + (num_edges / P);
}
for(i = 0; i < buckets; i++)
{
buckets_count[(t_id * buckets) + i] = 0;
}
for (i = offset_start; i < offset_end; i++)
{
u = (*edgeList)->edges_array_src[i];
t = (u >> (radix * 8)) & 0xff;
buckets_count[(t_id * buckets) + t]++;
}
#pragma omp barrier
if(t_id == 0)
{
for(i = 0; i < buckets; i++)
{
for(j = 0 ; j < P; j++)
{
t = buckets_count[(j * buckets) + i];
buckets_count[(j * buckets) + i] = base;
base += t;
}
}
}
#pragma omp barrier
for (i = offset_start; i < offset_end; i++)         
{
u = (*edgeList)->edges_array_src[i];
t = (u >> (radix * 8)) & 0xff;
o = buckets_count[(t_id * buckets) + t];
(*sorted_edges_array)->edges_array_dest[o] = (*edgeList)->edges_array_dest[i];
(*sorted_edges_array)->edges_array_src[o] = (*edgeList)->edges_array_src[i];
#if WEIGHTED
(*sorted_edges_array)->edges_array_weight[o] = (*edgeList)->edges_array_weight[i];
#endif
buckets_count[(t_id * buckets) + t]++;
}
}
temp_edges_array = *sorted_edges_array;
*sorted_edges_array = *edgeList;
*edgeList = temp_edges_array;
}
void radixSortCountSortEdgesByDestination (struct EdgeList **sorted_edges_array, struct EdgeList **edgeList, uint32_t radix, uint32_t buckets, uint32_t *buckets_count)
{
struct EdgeList *temp_edges_array = NULL;
uint32_t num_edges = (*edgeList)->num_edges;
uint32_t t = 0;
uint32_t o = 0;
uint32_t u = 0;
uint32_t i = 0;
uint32_t j = 0;
uint32_t P = 1;  
uint32_t t_id = 0;
uint32_t offset_start = 0;
uint32_t offset_end = 0;
uint32_t base = 0;
#pragma omp parallel default(none) shared(sorted_edges_array,edgeList,radix,buckets,buckets_count,num_edges) firstprivate(t_id, P, offset_end,offset_start,base,i,j,t,u,o)
{
P = omp_get_num_threads();
t_id = omp_get_thread_num();
offset_start = t_id * (num_edges / P);
if(t_id == (P - 1))
{
offset_end = offset_start + (num_edges / P) + (num_edges % P) ;
}
else
{
offset_end = offset_start + (num_edges / P);
}
for(i = 0; i < buckets; i++)
{
buckets_count[(t_id * buckets) + i] = 0;
}
for (i = offset_start; i < offset_end; i++)
{
u = (*edgeList)->edges_array_dest[i];
t = (u >> (radix * 8)) & 0xff;
buckets_count[(t_id * buckets) + t]++;
}
#pragma omp barrier
if(t_id == 0)
{
for(i = 0; i < buckets; i++)
{
for(j = 0 ; j < P; j++)
{
t = buckets_count[(j * buckets) + i];
buckets_count[(j * buckets) + i] = base;
base += t;
}
}
}
#pragma omp barrier
for (i = offset_start; i < offset_end; i++)         
{
u = (*edgeList)->edges_array_dest[i];
t = (u >> (radix * 8)) & 0xff;
o = buckets_count[(t_id * buckets) + t];
(*sorted_edges_array)->edges_array_dest[o] = (*edgeList)->edges_array_dest[i];
(*sorted_edges_array)->edges_array_src[o] = (*edgeList)->edges_array_src[i];
#if WEIGHTED
(*sorted_edges_array)->edges_array_weight[o] = (*edgeList)->edges_array_weight[i];
#endif
buckets_count[(t_id * buckets) + t]++;
}
}
temp_edges_array = *sorted_edges_array;
*sorted_edges_array = *edgeList;
*edgeList = temp_edges_array;
}
struct EdgeList *radixSortEdgesBySource (struct EdgeList *edgeList)
{
uint32_t radix = 4;  
uint32_t P = 1;  
uint32_t buckets = 256; 
uint32_t num_edges = edgeList->num_edges;
uint32_t *buckets_count = NULL;
uint32_t j = 0; 
struct EdgeList *sorted_edges_array = newEdgeList(num_edges);
sorted_edges_array->num_vertices = edgeList->num_vertices;
#pragma omp parallel default(none) shared(P,buckets,buckets_count)
{
uint32_t t_id = omp_get_thread_num();
if(t_id == 0)
{
P = omp_get_num_threads();
buckets_count = (uint32_t *) my_malloc(P * buckets * sizeof(uint32_t));
}
}
for(j = 0 ; j < radix ; j++)
{
radixSortCountSortEdgesBySource (&sorted_edges_array, &edgeList, j, buckets, buckets_count);
}
free(buckets_count);
freeEdgeList(sorted_edges_array);
return edgeList;
}
struct EdgeList *radixSortEdgesBySourceAndDestination (struct EdgeList *edgeList)
{
uint32_t radix = 4;  
uint32_t P = 1;  
uint32_t buckets = 256; 
uint32_t num_edges = edgeList->num_edges;
uint32_t *buckets_count = NULL;
uint32_t j = 0; 
struct EdgeList *sorted_edges_array = newEdgeList(num_edges);
sorted_edges_array->num_vertices = edgeList->num_vertices;
buckets_count = NULL;
#pragma omp parallel default(none) shared(P,buckets,buckets_count)
{
uint32_t t_id = omp_get_thread_num();
if(t_id == 0)
{
P = omp_get_num_threads();
buckets_count = (uint32_t *) my_malloc(P * buckets * sizeof(uint32_t));
}
}
for(j = 0 ; j < radix ; j++)
{
radixSortCountSortEdgesByDestination (&sorted_edges_array, &edgeList, j, buckets, buckets_count);
}
for(j = 0 ; j < radix ; j++)
{
radixSortCountSortEdgesBySource (&sorted_edges_array, &edgeList, j, buckets, buckets_count);
}
free(buckets_count);
freeEdgeList(sorted_edges_array);
return edgeList;
}
