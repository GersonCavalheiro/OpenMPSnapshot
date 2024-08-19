

#include "msort.hpp"


int* slice(int *arr, int start, int end)
{
try{
int *result = (int *) malloc((end - start) * sizeof(int));
int i;
for (i = start; i < end; i++)
{
result[i - start] = arr[i];
}
return result;
} catch (int e) {
printf("[Info] msort_slice Exception (%d).\n", e);
exit(EXIT_FAILURE);
}
}


void merge(
int *I, int *J,             
int *left_I, int *left_J,   
int *right_I, int *right_J, 
int leftLen, int rightLen)  
{

try{
int i, j;

i = 0;
j = 0;
while(i < leftLen && j < rightLen)
{
if (left_I[i] < right_I[j] ||
(left_I[i] == right_I[j] && left_J[i] < right_J[j]) 
)
{
I[i + j] = left_I[i];
J[i + j] = left_J[i];

i++;
}
else
{
I[i + j] = right_I[j];
J[i + j] = right_J[j];

j++;
}

}

for(; i < leftLen; i++)
{
I[i + j] = left_I[i];
J[i + j] = left_J[i];
}
for(; j < rightLen; j++)
{
I[i + j] = right_I[j];
J[i + j] = right_J[j];
}

free(left_I);
free(right_I);
free(left_J);
free(right_J);

} catch (int e) {
printf("[Info] msort_merge Exception (%d).\n", e);
exit(EXIT_FAILURE);
}
}

void mergeSort(int *I, int *J, int len)
{
try{
if (len <= 1)
{
return;
}

int *left_I = slice(I, 0, len / 2 + 1);
int *right_I = slice(I, len / 2, len);

int *left_J = slice(J, 0, len / 2 + 1);
int *right_J = slice(J, len / 2, len);

#pragma omp parallel
#pragma omp single
{
#pragma omp task
mergeSort(left_I, left_J,  len / 2);

#pragma omp task
mergeSort(right_I, right_J,  len - (len / 2));
}

merge(I, J,  left_I, left_J,  right_I, right_J,  len / 2, len - (len / 2));
} catch (int e) {
printf("[Info] msort_mergeSort Exception (%d).\n", e);
exit(EXIT_FAILURE);
}
}



void switcharoo_to_lower_triangle(int *I, int *J, int nz){

int t;

for(int i=0; i<nz; i++){
if(J[i] > I[i])
{
t = I[i];
I[i] = J[i];
J[i] = t;
}
}

}
