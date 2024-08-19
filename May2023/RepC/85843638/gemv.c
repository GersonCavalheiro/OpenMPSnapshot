#include "mnblas.h"
#include <stdio.h>
#include <nmmintrin.h>
#define VEC_SIZE 4
typedef struct {
float REEL;
float IMAG;
}vcomplexe;
typedef vcomplexe VCOMP [VEC_SIZE] ;
typedef struct {
double REEL;
double IMAG;
}dcomplexe;
typedef dcomplexe DCOMP [VEC_SIZE] ;
typedef float float4 [4] __attribute__ ((aligned (16))) ;
typedef double double2 [2] __attribute__ ((aligned (16))) ;
typedef vcomplexe matrix [4][4] ;
void printvec(float v[], int size){
for(int i = 0 ; i<size; i +=1)
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
void mncblas_sgemv_vec (const MNCBLAS_LAYOUT layout,
const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const float alpha, const float *A, const int lda,
const float *X, const int incX, const float beta,
float *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register float r ;
register float x ;
register unsigned int indice ;
float4 x4, r4 ;
__m128 xv4, a1, dot ;
for (i = 0; i < M; i += incX)
{
r = 0.0 ;
indice = i * M ;
x4 [0] = X [i] ;
x4 [1] = X [i] ;
x4 [2] = X [i] ;
x4 [3] = X [i] ;
xv4 = _mm_load_ps (x4) ;
for (j = 0 ; j < M; j += 4)
{
a1 = _mm_load_ps (A+indice+j) ;
dot = _mm_dp_ps (a1, xv4, 0xFF) ;
_mm_store_ps (r4, dot) ;
r += r4 [0] ;
}
Y [i] = (beta * Y[i])  + (alpha * r) ;
}
return ;
}
void mncblas_sgemv (const MNCBLAS_LAYOUT layout,
const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const float alpha, const float *A, const int lda,
const float *X, const int incX, const float beta,
float *Y, const int incY
)
{
register unsigned int i ;
register unsigned int j ;
register float r ;
register float x ;
register unsigned int indice ;
for (i = 0; i < M; i += incX)
{
r = 0.0 ;
x = X [i] ;
indice = i * M ;
for (j = 0 ; j < M; j += incY)
{
r += A[indice+j] * x ;
}
Y [i] = (beta * Y[i])  + (alpha * r) ;
}
return ;
}
void mncblas_sgemv_omp (const MNCBLAS_LAYOUT layout,
const MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const float alpha, const float *A, const int lda,
const float *X, const int incX, const float beta,
float *Y, const int incY
)
{
;
register unsigned int j ;
register float r ;
register float x ;
register unsigned int indice ;
#pragma omp for schedule(static) private(j, r, x, indice)
for (register unsigned int i = 0; i < M; i += incX)
{
r = 0.0 ;
x = X [i] ;
indice = i * M ;
for (j = 0 ; j < M; j += incY)
{
r += A[indice+j] * x ;
}
Y [i] = (beta * Y[i])  + (alpha * r) ;
}
return ;
}
void mncblas_dgemv (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const double alpha, const double *A, const int lda,
const double *X, const int incX, const double beta,
double *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register double r ;
register double x ;
register unsigned int indice ;
for (i = 0; i < M; i += incX)
{
r = 0.0 ;
x = X [i] ;
indice = i * M ;
for (j = 0 ; j < M; j += incY)
{
r += A[indice+j] * x ;
}
Y [i] = (beta * Y[i])  + (alpha * r) ;
}
return ;
}
void mncblas_dgemv_omp (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const double alpha, const double *A, const int lda,
const double *X, const int incX, const double beta,
double *Y, const int incY)
{
register unsigned int j ;
register double r ;
register double x ;
register unsigned int indice ;
#pragma omp for schedule(static) private(j, r, x, indice)
for (register unsigned int i = 0;i< M; i += incX)
{
r = 0.0 ;
x = X [i] ;
indice = i * M ;
for (j = 0 ; j < M; j += incY)
{
r += A[indice+j] * x ;
}
Y [i] = (beta * Y[i])  + (alpha * r) ;
}
return ;
}
void mncblas_dgemv_vec (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const double alpha, const double *A, const int lda,
const double *X, const int incX, const double beta,
double *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register double r ;
register double x ;
register unsigned int indice ;
double2 x4, r4 ;
__m128d xv4, a1, dot ;
for (i = 0; i < M; i += incX)
{
r = 0.0 ;
indice = i * M ;
x4 [0] = X [i] ;
x4 [1] = X [i] ;
xv4 = _mm_load_pd (x4) ;
for (j = 0 ; j < M; j += 2)
{
a1 = _mm_load_pd (A+indice+j) ;
dot = _mm_dp_pd (a1, xv4, 0xFF) ;
_mm_store_pd (r4, dot) ;
r += r4 [0] ;
}
Y [i] = (beta * Y[i])  + (alpha * r) ;
}
return ;
}
void mncblas_cgemv (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const void *alpha, const void *A, const int lda,
const void *X, const int incX, const void *beta,
void *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register vcomplexe r ;
register unsigned int indice ;
float *XP = (float *)X;
float *YP = (float *)Y;
float *AP = (float *)A;
vcomplexe *bv = (vcomplexe *)beta;
vcomplexe *av = (vcomplexe *)alpha;
for (i = 0; i < M*2; i += incX + 1)
{
r.REEL = 0; r.IMAG = 0;
indice = i * M ;
for (j = 0 ; j < M*2; j += incY + 1)
{
r.IMAG += (AP[indice+j] * *(XP+i+1)) + (*(XP+i) * AP[indice+j+1]) ;
r.REEL += (AP[indice+j] * *(XP+i)) - (AP[indice+j+1] * *(XP+i+1));
printf("-> %f %f \n", r.REEL, r.IMAG);
}
vcomplexe temp;
temp.REEL = *(YP+i);
temp.IMAG = *(YP+i+1);
*(YP+i) = ((temp.REEL * bv->REEL) - (temp.IMAG * bv->IMAG)) + ((av->REEL * r.REEL) - (av->IMAG * r.IMAG));
*(YP+i+1) = ((temp.REEL * bv->IMAG) + (temp.IMAG * bv->REEL)) + ((av->REEL * r.IMAG) + (av->IMAG * r.REEL));
}
return ;
}
void mncblas_cgemv_omp (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const void *alpha, const void *A, const int lda,
const void *X, const int incX, const void *beta,
void *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register vcomplexe r ;
register unsigned int indice ;
float *XP = (float *)X;
float *YP = (float *)Y;
float *AP = (float *)A;
vcomplexe *bv = (vcomplexe *)beta;
vcomplexe *av = (vcomplexe *)alpha;
#pragma omp for schedule(static) private(i, j, r, indice)
for (i = 0; i < M*2; i += incX + 1)
{
r.REEL = 0; r.IMAG = 0;
indice = i * M ;
for (j = 0 ; j < M*2; j += incY + 1)
{
r.IMAG += (AP[indice+j] * *(XP+i+1)) + (*(XP+i) * AP[indice+j+1]) ;
r.REEL += (AP[indice+j] * *(XP+i)) - (AP[indice+j+1] * *(XP+i+1));
}
vcomplexe temp;
temp.REEL = *(YP+i);
temp.IMAG = *(YP+i+1);
*(YP+i) = ((temp.REEL * bv->REEL) - (temp.IMAG * bv->IMAG)) + ((av->REEL * r.REEL) - (av->IMAG * r.IMAG));
*(YP+i+1) = ((temp.REEL * bv->IMAG) + (temp.IMAG * bv->REEL)) + ((av->REEL * r.IMAG) + (av->IMAG * r.REEL));
}
return ;
}
void mncblas_zgemv (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const void *alpha, const void *A, const int lda,
const void *X, const int incX, const void *beta,
void *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register dcomplexe r ;
register unsigned int indice ;
double *XP = (double *)X;
double *YP = (double *)Y;
double *AP = (double *)A;
dcomplexe *bv = (dcomplexe *)beta;
dcomplexe *av = (dcomplexe *)alpha;
for (i = 0; i < M*2; i += incX + 1)
{
r.REEL = 0; r.IMAG = 0;
indice = i * M ;
for (j = 0 ; j < M*2; j += incY + 1)
{
r.IMAG += (AP[indice+j] * *(XP+i+1)) + (*(XP+i) * AP[indice+j+1]) ;
r.REEL += (AP[indice+j] * *(XP+i)) - (AP[indice+j+1] * *(XP+i+1));
}
dcomplexe temp;
temp.REEL = *(YP+i);
temp.IMAG = *(YP+i+1);
*(YP+i) = ((temp.REEL * bv->REEL) - (temp.IMAG * bv->IMAG)) + ((av->REEL * r.REEL) - (av->IMAG * r.IMAG));
*(YP+i+1) = ((temp.REEL * bv->IMAG) + (temp.IMAG * bv->REEL)) + ((av->REEL * r.IMAG) + (av->IMAG * r.REEL));
}
return ;
}
void mncblas_zgemv_omp(MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const void *alpha, const void *A, const int lda,
const void *X, const int incX, const void *beta,
void *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register dcomplexe r ;
register unsigned int indice ;
double *XP = (double *)X;
double *YP = (double *)Y;
double *AP = (double *)A;
dcomplexe *bv = (dcomplexe *)beta;
dcomplexe *av = (dcomplexe *)alpha;
#pragma omp for schedule(static) private(i, j, r, indice)
for (i = 0; i < M*2; i += incX + 1)
{
r.REEL = 0; r.IMAG = 0;
indice = i * M ;
for (j = 0 ; j < M*2; j += incY + 1)
{
r.IMAG += (AP[indice+j] * *(XP+i+1)) + (*(XP+i) * AP[indice+j+1]) ;
r.REEL += (AP[indice+j] * *(XP+i)) - (AP[indice+j+1] * *(XP+i+1));
}
dcomplexe temp;
temp.REEL = *(YP+i);
temp.IMAG = *(YP+i+1);
*(YP+i) = ((temp.REEL * bv->REEL) - (temp.IMAG * bv->IMAG)) + ((av->REEL * r.REEL) - (av->IMAG * r.IMAG));
*(YP+i+1) = ((temp.REEL * bv->IMAG) + (temp.IMAG * bv->REEL)) + ((av->REEL * r.IMAG) + (av->IMAG * r.REEL));
}
return ;
}
void mncblas_cgemv_vec (MNCBLAS_LAYOUT layout,
MNCBLAS_TRANSPOSE TransA, const int M, const int N,
const void *alpha, const void *A, const int lda,
const void *X, const int incX, const void *beta,
void *Y, const int incY)
{
register unsigned int i ;
register unsigned int j ;
register vcomplexe r ;
register unsigned int indice ;
float *XP = (float *)X;
float *YP = (float *)Y;
float *AP = (float *)A;
vcomplexe *bv = (vcomplexe *)beta;
vcomplexe *av = (vcomplexe *)alpha;
for (i = 0; i < M*2; i += incX + 1)
{
r.REEL = 0; r.IMAG = 0;
indice = i * M ;
for (j = 0 ; j < M*2; j += incY + 1)
{
float4 o1, o2, rf;
o1[0] = o1[1] = *(AP+indice+j);
o1[2] = o1[3] = *(AP+indice+j+1);
o2[0] = *(XP+i); o2[3] = *(XP+i);
o2[2] = o2[1] = *(XP+i+1);
__m128 m1 = _mm_load_ps(o1);
__m128 m2 = _mm_load_ps(o2);
__m128 rmat = _mm_mul_ps(m1, m2);
__m128 rm = _mm_addsub_ps(rmat, _mm_shuffle_ps(rmat, rmat, _MM_SHUFFLE(0, 0, 3, 2)));
_mm_store_ps(rf, rm);
r.IMAG += rf[1];
r.REEL += rf[0];
}
vcomplexe temp;
temp.REEL = *(YP+i);
temp.IMAG = *(YP+i+1);
__m128 o1, o2, o3, o4, m1, m2, reel, imag;
float4 o1f, o2f, o3f, o4f, reel4, imag4;
o1f[0] = *(YP+i); o1f[1] = *(YP+i+1); o1f[2] = av->REEL; o1f[3] = av->IMAG;
o2f[0] = bv->IMAG; o2f[1] = bv->REEL; o2f[2] = r.IMAG; o2f[3] = r.REEL;
printvec(o1f, 4);
printvec(o2f, 4);
o3f[0] = o1f[0]; o3f[1] = o1f[1]; o3f[2] = av->REEL; o3f[3] = av->IMAG;
o4f[0] = bv->REEL; o4f[1] = bv->IMAG; o4f[2] = r.REEL; o4f[3] = r.IMAG;
printvec(o3f, 4);
printvec(o4f, 4);
o1 = _mm_load_ps(o1f);
o2 = _mm_load_ps(o2f);
o3 = _mm_load_ps(o3f);
o4 = _mm_load_ps(o4f);
m1 = _mm_mul_ps(o3, o4);
m2 = _mm_sub_ps(m1, _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(0, 3, 0, 1)));
_mm_store_ps(reel4, m2);
*(YP+i) = reel4[0] + reel4[2];
imag = _mm_dp_ps(o1, o2, 0xFF);
_mm_store_ps (imag4, imag) ;
*(YP+i+1) = imag4 [0] ;
}
return ;
}
int main(){
matrix M = {
{{1, 2}, {1, 2}, {1, 2}, {1, 2}},
{{1, 2}, {1, 2}, {1, 2}, {1, 2}},
{{1, 2}, {1, 2}, {1, 2}, {1, 2}},
{{1, 2}, {1, 2}, {1, 2}, {1, 2}}
};
vcomplexe y[4] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
vcomplexe y2[4] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
vcomplexe x[4] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}};
vcomplexe a = {1, 2};
vcomplexe b = {1, 2};
mncblas_cgemv_vec(101, 111, 4, 4, &a, &M, 0, &x, 1, &b, &y, 1);
printvec2(y);
return 0;
}