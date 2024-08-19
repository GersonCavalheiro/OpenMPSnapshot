#include "mnblas.h"
#include <stdio.h>
#include <x86intrin.h>
#define VEC_SIZE 4
typedef struct {
float REEL;
float IMAG;
} vcomplexe;
typedef vcomplexe VCOMP [VEC_SIZE] ;
typedef struct {
double REEL;
double IMAG;
} dcomplexe;
typedef dcomplexe DCOMP [VEC_SIZE] ;
#define NUM_PROC 2
#define NUM_THREADS 2
typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;
void printvec(float v[], int size){
for(int i = 0 ; i<size; i++)
printf("%f ", v[i]);
printf("\n");
}
void printvec2(VCOMP v){
for(int i = 0 ; i< VEC_SIZE; i++){
vcomplexe cc = v[i];
printf("(%f, %f) ", cc.REEL, cc.IMAG);
}
printf("\n");
}
void printvec3(DCOMP v){
for(int i = 0 ; i< VEC_SIZE; i++){
dcomplexe cc = v[i];
printf("(%f, %f) ", cc.REEL, cc.IMAG);
}
printf("\n");
}
void mncblas_saxpy_vec (const int N, const float alpha, const float *X,
const int incX, float *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
float4 alpha4 ;
__m128 x1, x2, y1, y2 ;
__m128 alpha1;
alpha4 [0] = alpha;
alpha4 [1] = alpha;
alpha4 [2] = alpha;
alpha4 [3] = alpha;
alpha1 = _mm_load_ps (alpha4) ;
for (i = 0, j = 0 ; j < N; i += 4, j += 4)
{
x1 = _mm_load_ps (X+i) ;
y1 = _mm_load_ps (Y+i) ;
x2 = _mm_mul_ps (x1, alpha1) ;
y2 = _mm_add_ps (y1, x2) ;
_mm_store_ps (Y+i, y2) ;
}
return ;
}
void mncblas_saxpy_omp (const int N, const float alpha, const float *X,
const int incX, float *Y, const int incY)
{
register unsigned int j ;
#pragma omp for schedule(static) private(j)
for (j = 0 ; j < N;j += 4)
{
Y [j] = alpha * X[j] + Y[j] ;
Y [j+1] = alpha * X[j+1] + Y[j+1] ;
Y [j+2] = alpha * X[j+2] + Y[j+2] ;
Y [j+3] = alpha * X[j+3] + Y[j+3] ;
}
return ;
}
void mncblas_saxpy (const int N, const float alpha, const float *X,
const int incX, float *Y, const int incY)
{
register unsigned int j ;
for (j = 0 ; j < N; j += 4)
{
Y [j] = alpha * X[j] + Y[j] ;
Y [j+1] = alpha * X[j+1] + Y[j+1] ;
Y [j+2] = alpha * X[j+2] + Y[j+2] ;
Y [j+3] = alpha * X[j+3] + Y[j+3] ;
}
return ;
}
void mncblas_daxpy_vec(const int N, const double alpha, const double *X,
const int incX, double *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
double2 alpha2 ;
__m128d x1, x2, y1, y2 ;
__m128d alpha1;
alpha2 [0] = alpha;
alpha2 [1] = alpha;
alpha1 = _mm_load_pd (alpha2) ;
for (i = 0, j = 0 ; j < N; i += 2, j += 2)
{
x1 = _mm_load_pd (X+i) ;
y1 = _mm_load_pd (Y+i) ;
x2 = _mm_mul_pd (x1, alpha1) ;
y2 = _mm_add_pd (y1, x2) ;
_mm_store_pd (Y+i, y2) ;
}
return ;
}
void mncblas_daxpy_omp (const int N, const double alpha, const double *X,
const int incX, double *Y, const int incY)
{
register unsigned int j ;
#pragma omp for schedule(static) private(j)
for (j = 0 ; j < N; j += 4)
{
Y [j] = alpha * X[j] + Y[j] ; 
Y [j+1] = alpha * X[j+1] + Y[j+1] ;
Y [j+2] = alpha * X[j+2] + Y[j+2] ;
Y [j+3] = alpha * X[j+3] + Y[j+3] ;   
}
return ;
}
void mncblas_daxpy (const int N, const double alpha, const double *X,
const int incX, double *Y, const int incY)
{
register unsigned int j ;
for (j = 0 ; j < N; j += 4)
{
Y [j] = alpha * X[j] + Y[j] ;
Y [j+1] = alpha * X[j+1] + Y[j+1] ;
Y [j+2] = alpha * X[j+2] + Y[j+2] ;
Y [j+3] = alpha * X[j+3] + Y[j+3] ;
}
return ;
}
void mncblas_caxpy(const int N, const void *alpha, const void *X,
const int incX, void *Y, const int incY)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
float *XP = (float *) X;
float *YP = (float *) Y;
float *AP = (float *) alpha;
vcomplexe temp;
for (; i < N*2 ; i += incX +2){
temp.REEL = (AP[0] * *(XP+i)) - (AP[1] * *(XP+i+1));
temp.IMAG = (AP[0] * *(XP+i+1)) + (AP[1] * *(XP+i));
*(YP+i) = temp.REEL + YP[i];
*(YP+i+1) = temp.IMAG + *(YP+i+1);
}
return ;
}
void mncblas_caxpy_omp(const int N, const void *alpha, const void *X,
const int incX, void *Y, const int incY)
{
float *XP = (float *) X;
float *YP = (float *) Y;
float *AP = (float *) alpha;
vcomplexe temp;
#pragma omp for schedule(static) private(temp)
for (register unsigned int i = 0; i < N*2 ; i += incX +2){
temp.REEL = (AP[0] * *(XP+i)) - (AP[1] * *(XP+i+1));
temp.IMAG = (AP[0] * *(XP+i+1)) + (AP[1] * *(XP+i));
*(YP+i) = temp.REEL + YP[i];
*(YP+i+1) = temp.IMAG + *(YP+i+1);
}
return ;
}
void mncblas_caxpy_vec(const int N, const void *alpha, const void *X,
const int incX, void *Y, const int incY)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
float *XP = (float *) X;
float *YP = (float *) Y;
float *AP = (float *) alpha;
float4 t;
__m128 o1, o2, rm;
o1 = _mm_set_ps(AP[1], AP[1], AP[0], AP[0]);
for (; ((i < N*2) && (j < N*2)) ; i += incX + 2 , j+=incY + 2){
o2 = _mm_set_ps(*(XP+i), *(XP+i+1), *(XP+i+1), *(XP+i));
rm = _mm_mul_ps(o1, o2);
rm = _mm_addsub_ps(rm, _mm_shuffle_ps(rm, rm, _MM_SHUFFLE(0, 0, 3, 2)));
_mm_store_ps(t, rm);
*(YP+j) += t[0];
*(YP+j+1) += t[1];
}
return ;
}
void mncblas_zaxpy(const int N, const void *alpha, const void *X,
const int incX, void *Y, const int incY)
{
register unsigned int i = 0 ;
double *XP = (double *) X;
double *YP = (double *) Y;
double *AP = (double *) alpha;
dcomplexe temp;
for (; i < N*2 ; i += incX +2){
temp.REEL = (AP[0] * *(XP+i)) - (AP[1] * *(XP+i+1));
temp.IMAG = (AP[0] * *(XP+i+1)) + (AP[1] * *(XP+i));
*(YP+i) = temp.REEL + YP[i];
*(YP+i+1) = temp.IMAG + *(YP+i+1);
}
}
void mncblas_zaxpy_omp(const int N, const void *alpha, const void *X,
const int incX, void *Y, const int incY)
{
double *XP = (double *) X;
double *YP = (double *) Y;
double *AP = (double *) alpha;
dcomplexe temp;
#pragma omp for schedule(static) private(temp)
for (register unsigned int i = 0; i < N*2 ; i += incX +2){
temp.REEL = (AP[0] * *(XP+i)) - (AP[1] * *(XP+i+1));
temp.IMAG = (AP[0] * *(XP+i+1)) + (AP[1] * *(XP+i));
*(YP+i) = temp.REEL + YP[i];
*(YP+i+1) = temp.IMAG + *(YP+i+1);
}
return ;
}
void mncblas_zaxpy_vec(const int N, const void *alpha, const void *X,
const int incX, void *Y, const int incY)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
double *XP = (double *) X;
double *YP = (double *) Y;
double *AP = (double *) alpha;
double2 t;
__m128d o11, o12, o21, o22, r1, r2, r3, yv;
o11 = _mm_set_pd(AP[0], AP[0]);
o12 = _mm_set_pd(AP[1], AP[1]);
for (; ((i < N*2) && (j < N*2)) ; i += incX +2 , j+=incY +2){
o21 = _mm_set_pd(*(XP+i+1), *(XP+i));
o22 = _mm_set_pd(*(XP+i), *(XP+i+1));
yv = _mm_set_pd(*(YP+j+1), *(YP+j));
r1 = _mm_mul_pd(o11, o21);
r2 = _mm_mul_pd(o12, o22);
r3 = _mm_addsub_pd(r1, r2);
r2 = _mm_add_pd(yv, r3);
_mm_store_pd(t, r2);
*(YP+j) = t[0];
*(YP+j+1) = t[1];
}
}
int main(){
VCOMP V1 = {{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}};
VCOMP V2 = {{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}};
vcomplexe a = {1.0, 2.0};
vcomplexe *p1 = &a;
mncblas_caxpy_vec(4, p1, V1, 0, V2, 0);
printvec2(V2);
}
