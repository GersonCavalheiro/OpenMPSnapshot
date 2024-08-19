#include <omp.h>									
#include <stdlib.h>									
#include "../common.h"								
#include "../util/timer/timer.h"					
#include "./kernel_cpu_2.h"							
void 
kernel_cpu_2(	int cores_arg,
knode *knodes,
long knodes_elem,
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
long long time0;
long long time1;
long long time2;
int i;
time0 = get_time();
int max_nthreads;
max_nthreads = omp_get_max_threads();
omp_set_num_threads(cores_arg);
int threadsPerBlock;
threadsPerBlock = order < 1024 ? order : 1024;
time1 = get_time();
int thid;
int bid;
#pragma omp parallel for private(i ,thid ) 
for(bid = 0; bid < count; bid++){
for(i = 0; i < maxheight; i++){
for(thid = 0; thid < threadsPerBlock; thid++){
if((knodes[currKnode[bid]].keys[thid] <= start[bid]) && (knodes[currKnode[bid]].keys[thid+1] > start[bid])){
if(knodes[currKnode[bid]].indices[thid] < knodes_elem){
offset[bid] = knodes[currKnode[bid]].indices[thid];
}
}
if((knodes[lastKnode[bid]].keys[thid] <= end[bid]) && (knodes[lastKnode[bid]].keys[thid+1] > end[bid])){
if(knodes[lastKnode[bid]].indices[thid] < knodes_elem){
offset_2[bid] = knodes[lastKnode[bid]].indices[thid];
}
}
}
currKnode[bid] = offset[bid];
lastKnode[bid] = offset_2[bid];
}
for(thid = 0; thid < threadsPerBlock; thid++){
if(knodes[currKnode[bid]].keys[thid] == start[bid]){
recstart[bid] = knodes[currKnode[bid]].indices[thid];
}
}
for(thid = 0; thid < threadsPerBlock; thid++){
if(knodes[lastKnode[bid]].keys[thid] == end[bid]){
reclength[bid] = knodes[lastKnode[bid]].indices[thid] - recstart[bid]+1;
}
}
}
time2 = get_time();
printf("Time spent in different stages of CPU/MCPU KERNEL:\n");
printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);
printf("Total time:\n");
printf("%.12f s\n", 												(float) (time2-time0) / 1000000);
} 
