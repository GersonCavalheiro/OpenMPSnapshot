#include <omp.h>									
#include <stdlib.h>									
#include <stdio.h>									
#include "../common.h"								
#include "../util/timer/timer.h"					
void 
kernel_cpu(	int cores_arg,
record *records,
knode *knodes,
long knodes_elem,
int order,
long maxheight,
int count,
long *currKnode,
long *offset,
int *keys,
record *ans)
{
long long time0;
long long time1;
long long time2;
time0 = get_time();
int max_nthreads;
max_nthreads = omp_get_max_threads();
omp_set_num_threads(cores_arg);
int threadsPerBlock;
threadsPerBlock = order < 1024 ? order : 1024;
time1 = get_time();
int thid;
int bid;
int i;
#pragma omp parallel for private(i ,thid ) 
for(bid = 0; bid < count; bid++){
for(i = 0; i < maxheight; i++){
for(thid = 0; thid < threadsPerBlock; thid++){
if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
if(knodes[offset[bid]].indices[thid] < knodes_elem){
offset[bid] = knodes[offset[bid]].indices[thid];
}
}
}
currKnode[bid] = offset[bid];
}
for(thid = 0; thid < threadsPerBlock; thid++){
if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
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
