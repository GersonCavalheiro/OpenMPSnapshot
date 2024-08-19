#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "../common.h"                
#include "../util/timer/timer.h"          
#include "./kernel2_wrapper.h"      


void 
kernel2_wrapper(
knode *knodes,
long knodes_elem,
long knodes_mem,  

int order,
long maxheight,
int count,

long *currKnode,
long *offset,
long *lastKnode,
long *offset_2,
int *start,
int *end,
int *recstart,
int *reclength)
{



size_t threads;
threads = order < 256 ? order : 256;

#pragma omp target data map(to: knodes[0: knodes_mem],\
start[0: count],\
end[0: count],\
currKnode[0: count],\
offset[0: count],\
lastKnode[0: count],\
offset_2[0: count])\
map(tofrom: recstart[0: count])\
map(from: reclength[0: count])
{
long long kernel_start = get_time();

#pragma omp target teams num_teams(count) thread_limit(threads)
{
#pragma omp parallel
{
int thid = omp_get_thread_num();
int bid = omp_get_team_num();

int i;
for(i = 0; i < maxheight; i++){

if((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid+1] > start[bid])){
if(knodes[currKnode[bid]].indices[thid] < knodes_elem) {
offset[bid] = knodes[currKnode[bid]].indices[thid];
}
}
if((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid+1] > end[bid])){
if(knodes[lastKnode[bid]].indices[thid] < knodes_elem) {
offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
}
}
#pragma omp barrier
if(thid==0){
currKnode[bid] = offset[bid];
lastKnode[bid] = offset_2[bid];
}
#pragma omp barrier
}

if(knodes[currKnode[bid]].keys[thid] == start[bid]){
recstart[bid] = knodes[currKnode[bid]].indices[thid];
}
#pragma omp barrier

if(knodes[lastKnode[bid]].keys[thid] == end[bid]){
reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid]+1;
}
}
}
long long kernel_end = get_time();
printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));
}

#ifdef DEBUG
for (int i = 0; i < count; i++)
printf("recstart[%d] = %d\n", i, recstart[i]);
for (int i = 0; i < count; i++)
printf("reclength[%d] = %d\n", i, reclength[i]);
#endif

}

