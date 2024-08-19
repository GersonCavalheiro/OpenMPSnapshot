#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "vector.h"
int allocate_Vector(Vector *v, int N){
v->N = N;
v->data = malloc(sizeof(float)*N);
if(v->data == NULL)
return 1;
memset(v->data, 0, sizeof(float)*N);
return 0;
}
int deallocate_Vector(Vector *v){
v->N = 0;
free(v->data);
v->data = NULL;
return 0;
}
int rand_fill_Vector(Vector *v){
srand(time(NULL));
if(v->N < 1)
return 1;
for(int i=0; i < v->N; i++){
v->data[i] = (float)rand() / (float)RAND_MAX;
}
return 0;
}
int zero_fill_Vector(Vector *v){
if(v->N < 1)
return 1;
memset(v->data, 0, sizeof(float)*v->N);
return 0;
}
float norm(Vector* v) {
int i;
int N = v->N;
float* v_data = v->data;
float length = 0.0;
#pragma omp parallel for private(i) shared(N, v_data) reduction(+:length)
for(i=0; i < N; i++){
length += pow(v_data[i], 2.0);
}
length = sqrt(length);
return length;
}
int normalize(Vector* v) {
int i;
int N = v->N;
float* v_data = v->data;
float length = norm(v);
if( abs(length) < 1e-5 ) return 0;
#pragma omp parallel for shared(N,v_data) private(i)
for(i=0; i < N; i++){
v_data[i] /= length;
}
return 0;
}
int axpy(float alpha, Vector* vx, Vector* vy, Vector* vz) {
int i;
int N = vx->N;
float* vx_data = vx->data;
float* vy_data = vy->data;
float* vz_data = vz->data;
if ((vx->N != vy->N) ||
(vx->N != vz->N)){
return 1;
}
#pragma omp parallel for shared(N,vz_data,vx_data,vy_data,alpha) private(i)
for(i=0; i < N; i++){
vz_data[i] = alpha*vx_data[i] + vy_data[i];
}
return 0;
}
int inner_product(Vector* vx, Vector* vy, float* ip) {
int i;
int N = vx->N;
float* vx_data = vx->data;
float* vy_data = vy->data;
float result = 0.0;
if (vx->N != vy->N) return 1;
#pragma omp parallel for shared(N,vx_data,vy_data) private(i) reduction(+:result)
for(i=0; i < N; i++){
result += vx_data[i]*vy_data[i];
}
*ip = result;
return 0;
}
