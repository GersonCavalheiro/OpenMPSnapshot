#include "psort.h"
#include <omp.h>



int get_partition( uint32_t *array, int small, int big){
uint32_t ele_at_pivot = array[big];
int new_small = small-1;
int start = new_small;
for (int j = small; j <= big-1; j++){
if (array[j] < 1 + ele_at_pivot){
int reverse = array[++start];
array[start] = array[j];
array[j] = reverse;
}
}
int r = start+1;
int rev = array[r];
array[r] = array[big];
array[big] = rev;
return r;
}


void SequentialSort(uint32_t *array, int small, int big, bool boolean){
if (big >= small+1 && boolean == true){
int parti = get_partition(array, small, big);
SequentialSort(array, parti + 1, big, true);
SequentialSort(array, small, parti - 1, true);
}
}

void ParallelSort(uint32_t *data, uint32_t n, int p)
{
int thres_hold = (2*(int)(n))/p;
uint32_t **all_buckets = new uint32_t*[p];
int i;
int size_bucket = (int)n/p;
int extra_elements = ((int)n%p);
for(i =0; i<extra_elements; i++){
uint32_t *large_bucket = new uint32_t[size_bucket +1];
all_buckets[i] = large_bucket;
}
for(i = extra_elements; i<p; i++){
uint32_t *small_bucket = new uint32_t[size_bucket];
all_buckets[i] = small_bucket;

}
int data_ind= 0;
for(int ir =0 ; ir<extra_elements; ir++){
for(int i2 =0; i2<size_bucket+1 ; i2++){
all_buckets[ir][i2] = data[data_ind];
data_ind++;
}
}
for(int ir2 =extra_elements ; ir2<p; ir2++){
for(int iit =0; iit<size_bucket; iit++){
all_buckets[ir2][iit] = data[data_ind];
data_ind++;
}
}
uint32_t *pseudo_splitters = new uint32_t[p*p];
int index=0;
for(int i =0; i<p; i++){
for(int j =0; j<p; j++){
pseudo_splitters[index]= all_buckets[i][j];
index++;
}

}   
SequentialSort(pseudo_splitters ,0 , (p*p-1), true);
uint32_t *p_1splitters = new uint32_t[p-1];
for(int ele =0 ;ele<=p-2; ele++){
p_1splitters[ele] = pseudo_splitters[((ele+1)*p)];
}
uint32_t **divided_tasks = new uint32_t*[p];
for(int t =0; t<p; t++){
uint32_t *per_task = new uint32_t[n];
divided_tasks[t] = per_task;
}
int *count_arr = new int[p];

for (int nl = 0; nl < p; nl++){
#pragma omp task firstprivate(nl)
{
int count_per =0;
if (nl==0){
uint32_t end = p_1splitters[0];
for(int c =0; c<(int)n; c++){
if(data[c] <=end){
divided_tasks[nl][count_per] = data[c];
count_per++;
}
}

}
else if(nl == p-1){
uint32_t start = p_1splitters[p-2];
for(int c =0; c<(int)n; c++){
if(data[c] > start){
divided_tasks[nl][count_per] = data[c];
count_per++;
}
}

}
else{
uint32_t start = p_1splitters[nl-1];
uint32_t end = p_1splitters[nl];
for(int c =0; c<(int)n; c++){
if(data[c] > start && data[c]<=end){
divided_tasks[nl][count_per] = data[c];
count_per++;
}
}
}
count_arr[nl] = count_per;
}
}
#pragma omp taskwait
for(int x = 0 ; x<p ; x++){
if (count_arr[x] < thres_hold){
SequentialSort(divided_tasks[x], 0, count_arr[x]-1, true);
}
else{
SequentialSort(divided_tasks[x], 0, count_arr[x]-1, false);
ParallelSort(divided_tasks[x], (uint32_t)count_arr[x], p);
}
}

int stream_ind =0;
for(int i =0; i<p; i++){
for(int j =0; j<count_arr[i];j++){
data[stream_ind] = divided_tasks[i][j];
stream_ind++;
}
}
}
