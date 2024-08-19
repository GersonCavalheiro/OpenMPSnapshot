#include<stdio.h>
#include<string.h>
#define ULLMAX (0ULL - 1ULL)
#define MAXN 2050
typedef unsigned long long ULL;
int N; 
ULL dimensions[MAXN]; 
ULL dptable[MAXN][MAXN]; 
ULL matrix_min_chain(ULL dimensions[]){
for(int i = 0; i < N; i++){
dptable[i][i] = 0; 
if(i <= N - 2){
dptable[i+1][i] = dptable[i][i+1] = dimensions[i] * dimensions[i+1] * dimensions[i+2];
}
}
#pragma omp parallel 
for(int length = 3; length <= N; length++){
const int start_point_boundary = N - length; 
#pragma omp for
for(int i = 0; i <= start_point_boundary; i++){
ULL minimum = ULLMAX;
ULL temp_sum;
const int split_point_boundary = i + length - 2; 
const int end_boundary = i + length - 1;
for(int j = i; j <= split_point_boundary; j++){
#ifdef CACHEMISS
temp_sum = dptable[i][j] + dimensions[i] * dimensions[j+1] * dimensions[i+length] + dptable[j+1][end_boundary];
#else
temp_sum = dptable[i][j] + dimensions[i] * dimensions[j+1] * dimensions[i+length] + dptable[end_boundary][j+1];
#endif
if(temp_sum < minimum){
minimum = temp_sum;
}
}
dptable[end_boundary][i] = dptable[i][end_boundary] = minimum;
}
}
return dptable[0][N-1];
}
int main(){
while(scanf("%d", &N) != EOF){
for(int i = 0; i < N+1; i++){
scanf("%llu", &dimensions[i]);
}
printf("%llu\n", matrix_min_chain(dimensions));
}
}
