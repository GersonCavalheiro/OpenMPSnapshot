#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#define N 100
#define C 16
int main(){
int var[N];
for(int i=0; i<N; i++){
var[i]=0;
}
#pragma omp target map(tofrom:var[0:N]) device(0)
#pragma omp teams distribute parallel for simd safelen(C)
for (int i=C; i<N; i++){
var[i]=var[i-C]+1;
}
printf("%d\n",var[97]);
return 0;
}
