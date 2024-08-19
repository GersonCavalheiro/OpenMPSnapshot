
#include "sparse.hpp"

void sparseSortMtx(CSCMatrix* mx){

#pragma omp parallel for
for(int i=0; i<mx->W; i++){
quicksort(mx->csci, mx->cscp[i], mx->cscp[i+1]-1);
}

}


void quicksort(int* arr,int first,int last){
int i, j, pivot, temp;

if(first<last){
pivot = first;
i = first;
j = last;

while(i<j){
while(arr[i]<=arr[pivot]&&i<last)
i++;
while(arr[j]>arr[pivot])
j--;
if(i<j){
temp=arr[i];
arr[i]=arr[j];
arr[j]=temp;
}
}

temp = arr[pivot];
arr[pivot] = arr[j];
arr[j] = temp;

#pragma omp parallel
#pragma omp single
{
#pragma omp task
quicksort(arr,first,j-1);

#pragma omp task
quicksort(arr,j+1,last);
}
}
}





