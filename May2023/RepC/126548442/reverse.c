#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "myRead.h"
struct tuple{
int csrRowIdx;
int csrCollIdx;
short weight;
};
typedef struct tuple tuple;
struct csr{
int n;  
int nnz; 
int *csrRowPtr; 
tuple** tuple; 
};
typedef struct csr csr;
csr* getCsr(int n, int nnz){
csr* Matrix = (csr*)malloc(sizeof(csr));
Matrix -> n = n;
Matrix -> nnz = nnz;
Matrix -> csrRowPtr = (int*) calloc(n+1,sizeof(int) );
Matrix -> tuple = (tuple**) calloc( nnz ,sizeof(tuple*));
return Matrix;
}
int* sum(int n, int* arr1, int* arr2){
int i = 0;
for(i = 0; i < n+1; i++){
arr1[i] += arr2[i];
}
return arr1;
}
int comparator (const void * elem1, const void * elem2)  
{
int f = ((tuple*)elem1)->csrCollIdx;
int s = ((tuple*)elem2)->csrCollIdx;
if (f > s) return  1;
if (f < s) return -1;
return 0;
}
tuple** insertInOrder(tuple** arr, int start, int size, int insert){
int i;
for(i = start; i < size + 1; i ++){
if(comparator(arr[insert], arr[i]) < 0){
break;
}
}
tuple* temp;
for(int j = i; j < size + 1; j ++){
temp = arr[j];
arr[j] = arr[insert];
arr[insert] = temp;
}
return arr;
}
tuple** insertionSort(tuple** arr, int start, int size){
for(int i = start; i < size - 1; i ++){
arr = insertInOrder(arr, start, i, i+1);
}
return arr;
}
void merge(csr* inMatrix, int start1, int mid, int end2)
{   
tuple** a = inMatrix->tuple;    
tuple** temp = (tuple**) malloc(sizeof(tuple*)*(end2-start1+1));    
int i,j,k;
i=start1;    
j=mid+1;    
k=0;
while(i<=mid && j<=end2)    
{
if(a[i]->csrCollIdx <= a[j]->csrCollIdx)
temp[k++]=a[i++];
else
temp[k++]=a[j++];
}
while(i<=mid)    
temp[k++]=a[i++];
while(j<=end2)    
temp[k++]=a[j++];
for(i=start1,j=0;i<=end2;i++,j++)
a[i]=temp[j];
free(temp);
}
int *mergeTrans(csr* inMatrix, int i, int j)
{
int mid;
int* arr1;
int* arr2;
if(i + 10000 < j)
{
mid=(int)(i+j)/2;
#pragma omp task shared(arr1, inMatrix,i,mid)
arr1 = mergeTrans(inMatrix,i,mid);        
#pragma omp task shared(arr2, inMatrix,j,mid)
arr2 = mergeTrans(inMatrix,mid+1,j);    
#pragma omp taskwait
{   merge(inMatrix, i,mid,j);    
arr1 =  sum( inMatrix->n, arr1, arr2 );
free(arr2);
return arr1;
}
}
else{
arr1 = (int*)calloc(inMatrix->n+1,sizeof(int));
int  k = 0, l = 0;
for(l = i; l <= j; l++){
for(k =inMatrix->tuple[l]-> csrCollIdx+1; k < inMatrix->n+1; k++){
arr1[k] += 1;
}
}   
inMatrix -> tuple = insertionSort(inMatrix -> tuple, i, j + 1);
return arr1;
}
}
int main(int argc, char **argv)
{   
int num_threads;
char* filename;
if(argc > 1) num_threads  = atoi(argv[1]);
else num_threads = 1;
if(argc > 2) filename = argv[2];
else{
filename = (char*) malloc(24 * sizeof(char));
memset(filename, '\0', 24);
strcpy(filename , "testcases/testcase.data");
}
int n, nnz;
int *csrRowPtr, *csrColIdx, *csrVal;
int *cscColPtr;
init_values(filename, &csrRowPtr, &csrColIdx, &csrVal, &n, &nnz);
csr* inMatrix = getCsr(n-1, nnz);
int i = 0, j = 0, k =0;
int chunk = 3;
inMatrix->csrRowPtr = csrRowPtr;
omp_set_num_threads(num_threads);
#pragma omp parallel for private(j)
for(j = 0; j < inMatrix->nnz; j++){
inMatrix->tuple[j] = (tuple*) calloc(1 ,sizeof(tuple));
inMatrix->tuple[j]-> csrRowIdx = 0 ;
inMatrix->tuple[j]-> csrCollIdx = csrColIdx[j];
inMatrix->tuple[j]-> weight = csrVal[j];
}
double starttime = omp_get_wtime();    
#pragma omp parallel for  private(k,j)
for(j = 1; j<inMatrix->n+1; j++){
for(k = inMatrix->csrRowPtr[j-1]; k < inMatrix->csrRowPtr[j]; k++ ){
inMatrix->tuple[k]-> csrRowIdx = j-1;        
}
}
#pragma omp parallel
{
#pragma omp single
cscColPtr = mergeTrans(inMatrix, 0, nnz-1);
}
printf("time taken %14.7f \n", (omp_get_wtime() - starttime));
for(j = 0; j<inMatrix->n; j++){
printf("%d ", cscColPtr[j] );
}
printf("%d", cscColPtr[j] );
printf("\n");       
for(j = 0; j<nnz-1; j++){
printf("%d ", inMatrix->tuple[j]-> csrRowIdx );
} 
printf("%d", inMatrix->tuple[j]-> csrRowIdx );
printf("\n");
for(j = 0; j<nnz-1; j++){
printf("%d ", inMatrix->tuple[j]->weight );
}
printf("%d", inMatrix->tuple[j]->weight );
printf("\n");
}
