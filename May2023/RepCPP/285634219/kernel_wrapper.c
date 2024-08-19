#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "../common.h"                
#include "../util/timer/timer.h"          
#include "./kernel_wrapper.h"      


void 
kernel_wrapper(  record *records,
long records_mem, 
knode *knodes,
long knodes_elem,
long knodes_mem,  

int order,
long maxheight,
int count,

long *currKnode,
long *offset,
int *keys,
record *ans)
{



int threads = order < 256 ? order : 256;

#pragma omp target data map(to: knodes[0: knodes_mem],\
records[0: records_mem],\
keys[0: count], \
currKnode[0: count],\
offset[0: count])\
map(from: ans[0: count])
{
long long kernel_start = get_time();

#pragma omp target teams num_teams(count) thread_limit(threads)
{
#pragma omp parallel
{
int thid = omp_get_thread_num();
int bid = omp_get_team_num();

for(int i = 0; i < maxheight; i++){

if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
if(knodes[offset[bid]].indices[thid] < knodes_elem){
offset[bid] = knodes[offset[bid]].indices[thid];
}
}
#pragma omp barrier
if(thid==0){
currKnode[bid] = offset[bid];
}
#pragma omp barrier
}

if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
}
}
}
long long kernel_end = get_time();
printf("Kernel execution time: %f (us)\n", (float)(kernel_end-kernel_start));
} 

#ifdef DEBUG
for (int i = 0; i < count; i++)
printf("ans[%d] = %d\n", i, ans[i].value);
printf("\n");
#endif

}

