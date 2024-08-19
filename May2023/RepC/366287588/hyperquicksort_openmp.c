#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>

#define NO_OF_THREADS 4

typedef struct ChunkInfo{
int start;
int end;
} ChunkInfo;

typedef struct OutgoingInfo{
int* arr;   
int to;
int arrSize;
bool isLow;
} OutgoingInfo;

int* arrRead;
int* arrWrite;
int medians[NO_OF_THREADS];
int chunkSizes[NO_OF_THREADS];
ChunkInfo infos[NO_OF_THREADS];
OutgoingInfo outInfos[NO_OF_THREADS];
int noOfElements;



void hyperquicksort2();
void quicksort(int start, int end);

int main(int argc, char** argv){

if(argc <= 2){
printf("Invalid Input.\n");
exit(0);
}

struct timeval time_seq_before, time_seq_after, time_final; 
gettimeofday(&time_seq_before, NULL);    
noOfElements = atoi(argv[1]);

int choice = 0 ;


choice = atoi(argv[2]);



arrRead = (int *)malloc(sizeof(int)*noOfElements);
arrWrite = (int *)malloc(sizeof(int)*noOfElements);
int *a3 = (int *)malloc(sizeof(int)*noOfElements);

omp_set_num_threads(NO_OF_THREADS);


for(int i = 0; i<noOfElements; i++){
arrWrite[i] = rand() % 100;
a3[i] = arrWrite[i];
}


if(choice){
printf("\nInitial Array: ");
for(int i = 0; i<noOfElements; i++){
printf("%d ", arrWrite[i]);
}
printf("\n");
}


int chunk = (noOfElements / NO_OF_THREADS);
int idx = 0;
for(int i = 0; i<NO_OF_THREADS; i++){
infos[i].start = idx;
if(i == (NO_OF_THREADS - 1))
infos[i].end = noOfElements-1;
else
infos[i].end = idx + chunk-1;

if(i != (NO_OF_THREADS - 1))
chunkSizes[i] = chunk;
else
chunkSizes[i] = noOfElements - idx; 

idx += chunk;
}
gettimeofday (&time_seq_after, NULL);    
printf("Sequential portion computation time: %ld microseconds\n", ((time_seq_after.tv_sec - time_seq_before.tv_sec)*1000000L + time_seq_after.tv_usec) - time_seq_before.tv_usec);
double x = omp_get_wtime();
quicksort(0, noOfElements-1);
double y = omp_get_wtime();
printf("Sequential quicksort timing: %lf seconds\n", (y-x));
arrWrite = a3;
double par_start = omp_get_wtime();
hyperquicksort2();
double par_end = omp_get_wtime();
printf("HYPER QUICK SORT ENDED\n");


printf("Time taken by hyperquicksort: %lf\n", par_end - par_start);

gettimeofday (&time_final, NULL);    
printf("\nTotal time: %ld microseconds\n", ((time_final.tv_sec - time_seq_before.tv_sec)*1000000L + time_final.tv_usec) - time_seq_before.tv_usec);

if(choice){
printf("\nFinal Array after sorting: ");
for(int i = 0; i<noOfElements; i++){
printf("%d ", arrWrite[i]);
}

printf("\n");

}

double hyperquicksortTime = par_end - par_start;
double quicksortTime = y - x;
double speedup = quicksortTime / hyperquicksortTime;
printf("\nSpeedup obtained = %lf", speedup);

return 0;
}

int partition(int start, int end){
int pivot = arrWrite[end];  
int i = (start - 1); 

for (int j = start; j <= end - 1; j++) 
{ 
if (arrWrite[j] < pivot) 
{ 
i++; 
int temp = arrWrite[i];
arrWrite[i] = arrWrite[j];
arrWrite[j] = temp;
} 
} 
int temp = arrWrite[i+1];
arrWrite[i+1] = arrWrite[end];
arrWrite[end] = temp;
return (i + 1); 
}

void quicksort(int start, int end){
if(start >= end) return;
if(start < end){
int pivot = partition(start, end);
quicksort(start, pivot-1);
quicksort(pivot+1, end);
}
}


void hyperquicksort2(){




for(int j=0; j<(int)log2(NO_OF_THREADS); j++){

#pragma omp parallel shared(infos, arrWrite, arrRead, medians, outInfos, noOfElements, chunkSizes, j) default(none)
{    
#pragma omp barrier
int currId = omp_get_thread_num();
if(j == 0){
int startIndex = infos[currId].start;
int endIndex = infos[currId].end; 
quicksort(startIndex, endIndex);

#pragma omp barrier
}


bool sendLow;
int neighbour;
int bitPosition = (1 << ((int)(log2(NO_OF_THREADS)) - j - 1));
if((currId & bitPosition)){
sendLow = true;
}
else{
sendLow = false;
}
neighbour = currId ^ bitPosition;

for(int k = infos[currId].start; k<=infos[currId].end; k++){
arrRead[k] = arrWrite[k];
}

#pragma omp barrier

bool execute = true;
if((currId << j) == 0){
if(infos[currId].start == -1){
if(infos[neighbour].start == -1){
execute = false;
}
else{
medians[currId] = arrRead[infos[neighbour].start + (infos[neighbour].end - infos[neighbour].start)/2];
}
}
else
medians[currId] = arrRead[infos[currId].start + (infos[currId].end - infos[currId].start)/2];
}
#pragma omp barrier

if(execute){

int t2 = INT_MAX << ((int)log2(NO_OF_THREADS) - j);
int medianIndex =  t2  & currId;    

int medianValue = medians[medianIndex];
int *tempArr = (int *)malloc(sizeof(int)*noOfElements);
int tempIdx = 0;
for(int k = infos[currId].start; k<=infos[currId].end;k++){
if(sendLow && (arrRead[k] <= medianValue)){
tempArr[tempIdx++] = arrRead[k]; 
arrRead[k] = -1;
}
else if(!sendLow && (arrRead[k] > medianValue)){
tempArr[tempIdx++] = arrRead[k]; 
arrRead[k] = -1;    
} 

}
outInfos[currId].arr = tempArr;
outInfos[currId].arrSize = tempIdx;
outInfos[currId].isLow = sendLow;
outInfos[currId].to = neighbour;


#pragma omp barrier



#pragma omp barrier
chunkSizes[currId] -= outInfos[currId].arrSize;
chunkSizes[currId] += outInfos[neighbour].arrSize;
#pragma omp barrier



int startIdx = 0;
for(int idx = 0; idx < currId; idx++) startIdx += chunkSizes[idx];

int idx1 = infos[currId].start, idx2 = 0, idx_write = startIdx;

int n1 = infos[currId].end+1;
int n2 = outInfos[neighbour].arrSize;

if(chunkSizes[currId] == 0){
infos[currId].start = -1;
infos[currId].end = -2;
}
else{
infos[currId].start = startIdx;
infos[currId].end = startIdx + chunkSizes[currId]-1;
}
#pragma omp barrier
while((idx1 < n1) && (idx2 < n2)){
if(arrRead[idx1] == -1){
idx1++;
}
else if(arrRead[idx1] >= outInfos[neighbour].arr[idx2]){
arrWrite[idx_write++] = outInfos[neighbour].arr[idx2];
idx2++;
}
else{
arrWrite[idx_write++] = arrRead[idx1];
idx1++;
}
}
if(idx1 < n1){
while(idx1 < n1){
if(arrRead[idx1] != -1)
arrWrite[idx_write++] = arrRead[idx1];
idx1++;    
}
}
if(idx2 < n2){

while(idx2 < n2){
arrWrite[idx_write++] = outInfos[neighbour].arr[idx2++]; 
}
}

}
#pragma omp barrier 
}
}

}

