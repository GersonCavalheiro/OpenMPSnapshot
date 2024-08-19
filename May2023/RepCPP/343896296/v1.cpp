

#include "v1.hpp"


void V1::Execute(Runtime rt){

if(!rt.opt_csr_a || rt.opt_csr_b || !rt.opt_csr_f){
printf("[Error] Flags '--opt-csr-a' and '--opt-csr-f' are required for V1.\n");
exit(EXIT_FAILURE);
}

time_t tim;
srand((unsigned) time(&tim));
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC_RAW, &start);

COOMatrix C = COOMatrix();
COOMatrix coo = COOMatrix();

#pragma omp parallel for \
private(coo) shared(C)
for(int j=0; j<rt.B->W; j++){

coo.Reset();


for(int i=0; i<rt.A->W; i++){

for(int k=rt.A->cscp[i]; k<rt.A->cscp[i+1]; k++){

if(!V1::binarySearch(rt.F->csci, rt.F->cscp[i], rt.F->cscp[i+1]-1, rt.A->csci[k]))
continue;

if(V1::binarySearch(rt.B->csci, rt.B->cscp[j], rt.B->cscp[j+1]-1, rt.A->csci[k])){

coo.addPoint(i, j);

break;

}

}

}

#pragma omp critical
{
C.coo.insert( C.coo.end(), coo.coo.begin(), coo.coo.end() );
C.nnz += coo.nnz;
}

}


clock_gettime(CLOCK_MONOTONIC_RAW, &end);
float delta_us = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000)/ (1000000);
printf("[Info] V1 took %f s\n", delta_us);


}


bool V1::binarySearch(int* arr, int l, int r, int x)
{

if (r >= l) {
int mid = l + (r - l) / 2;

if (arr[mid] == x)
return true;

if (arr[mid] > x)
return binarySearch(arr, l, mid - 1, x);

return binarySearch(arr, mid + 1, r, x);
}

return false;

}
