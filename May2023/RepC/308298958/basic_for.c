#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define PARALLELISM_ENABLED 1
#define M 800
#define N 800
#define P 800
int main(){
int A[M][N], B[N][P], C[M][P];
int i, j, k;
time_t start_t, end_t;
double total;
srand(time(NULL));
for(i = 0; i < M; i++){
for(j = 0; j < N; j++){
A[i][j] = 1 + rand() % 100; 
}
}
for(i = 0; i < N; i++){
for(j = 0; j < P; j++){
B[i][j] = 1 + rand() % 100; 
}
}
for(i = 0; i < M; i++){
for(j = 0; j < P; j++){
C[i][j] = 0;
}
}
start_t = time(NULL);
#pragma omp parallel for private(i, j, k) if(PARALLELISM_ENABLED)
for(i = 0; i < M; i++){
for(j = 0; j < P; j++){
for(k = 0; k < N; k++){
C[i][j] += A[i][k] * B[k][j];
}
}
}
end_t = time(NULL);
total = difftime(end_t, start_t);
printf("Loop execution took: %.2f seconds (parallelism enabled: %d)\n", total, PARALLELISM_ENABLED);
return 0;	
}
