#include "mnblas.h"
#include <x86intrin.h>
#include <emmintrin.h>
#include <stdio.h>
#define XMM_NUMBER 8
#define VEC_SIZE 5
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
void printvec(double v[], int size){
for(int i = 0 ; i<size; i++)
printf("%f ", v[i]);
printf("\n");
}
float mncblas_sdot(const int N, const float *X, const int incX, 
const float *Y, const int incY)
{
register unsigned int i = 0 ;
register float dot = 0.0 ;
for (; i < N ; i += incX)
{
dot = dot + X [i] * Y [i] ;
}
return dot ;
}
float mncblas_sdot_omp(const int N, const float *X, const int incX, 
const float *Y, const int incY)
{
register unsigned int i = 0 ;
register float dot = 0.0 ;
#pragma omp parallel for schedule(static) private(i) reduction(+:dot) 
for (i = 0; i < N ; i += incX)
{
dot = dot + X [i] * Y [i] ;
}
return dot ;
}
float mncblas_sdot_vec(const int N, const float *X, const int incX, 
const float *Y, const int incY)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
__m128 dot = _mm_set1_ps(0.0);
float4 fdot;
for (; ((i < N) && (j < N)) ; i += incX + 4, j+=incY + 4)
{
dot = _mm_add_ps(dot, _mm_mul_ps (_mm_load_ps (X+i), _mm_load_ps (Y+i)));
}
_mm_store_ps(fdot, dot);
return (fdot[0] + fdot[1] + fdot[2] + fdot[3]) ;
}
double mncblas_ddot(const int N, const double *X, const int incX, 
const double *Y, const int incY)
{
register unsigned int i = 0 ;
register double dot = 0.0 ;
for (; i < N ; i += incX)
{
dot = dot + X [i] * Y [i] ;
}
return dot ;
}
double mncblas_ddot_omp(const int N, const double *X, const int incX, 
const double *Y, const int incY)
{
register unsigned int i = 0 ;
register double dot = 0.0 ;
#pragma omp parallel for schedule(static) reduction(+:dot) private(i)
for (i=0 ; i < N ; i += incX)
{
dot = dot + X [i] * Y [i] ;
}
return dot ;
}
double mncblas_ddot_vec(const int N, const double *X, const int incX, 
const double *Y, const int incY)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
__m128d dot = _mm_set1_pd(0.0);
double2 fdot;
for (; ((i < N) && (j < N)) ; i += incX + 2, j+=incY + 2)
{
dot = _mm_add_pd(dot, _mm_mul_pd (_mm_load_pd (X+i), _mm_load_pd (Y+i)));
}
_mm_store_pd(fdot, dot);
return (fdot[0] + fdot[1]) ;
}
void mncblas_cdotu_sub(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotu)
{
register unsigned int i = 0 ;
float *XP = (float *)X;
float *YP = (float *)Y;
vcomplexe *temp = (vcomplexe *)dotu;
temp->IMAG = 0;
temp->REEL = 0;
for (; ((i < N*2)) ; i += incX + 2){
temp->REEL = temp->REEL + ((*(XP+i) * *(YP+i)) - (*(XP+i+1) * *(YP+i+1)));
temp->IMAG = temp->IMAG + ((*(XP+i) * *(YP+i+1)) + (*(XP+i+1) * *(YP+i)));
}
}
void mncblas_cdotu_sub_omp(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotu)
{
float *XP = (float *)X;
float *YP = (float *)Y;
vcomplexe *temp = (vcomplexe *)dotu;
float re = 0; float im = 0;
#pragma omp parallel for schedule(static) reduction(+:re, im)
for (register unsigned int i = 0; i < N*2; i += incX + 2){
re = re + ((*(XP+i) * *(YP+i)) - (*(XP+i+1) * *(YP+i+1)));
im = im + ((*(XP+i) * *(YP+i+1)) + (*(XP+i+1) * *(YP+i)));
}
temp->REEL = re;
temp->IMAG = im;
}
void mncblas_cdotu_sub_vec(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotu)
{
register unsigned int i = 0 ;
float *XP = (float *)X;
float *YP = (float *)Y;
vcomplexe *temp = (vcomplexe *)dotu;
temp->IMAG = 0;
temp->REEL = 0;
__m128 v1, v2, resm;
float4 _v1, _v2, resf;
for (; ((i < N*2)) ; i += incX + 2){
_v2[3] = _v1[0] = *(XP+i);
_v2[2] = _v1[1] = *(XP+i+1);
_v2[0] = _v1[2] = *(YP+i);
_v2[1] = _v1[3] = *(YP+i+1);
v1 = _mm_load_ps(_v1);
v2 = _mm_load_ps(_v2);
resm = _mm_mul_ps(v1, v2);
_mm_store_ps(resf, resm);
temp->REEL = temp->REEL + (resf[0] - resf[1]);
temp->IMAG = temp->IMAG + (resf[3] + resf[2]);
}
}
void mncblas_zdotu_sub(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotu)
{
register unsigned int i = 0 ;
double *XP = (double *)X;
double *YP = (double *)Y;
dcomplexe *temp = (dcomplexe *)dotu;
temp->IMAG = 0;
temp->REEL = 0;
for (; ((i < N*2)) ; i += incX + 2){
temp->REEL = temp->REEL + ((*(XP+i) * *(YP+i)) - (*(XP+i+1) * *(YP+i+1)));
temp->IMAG = temp->IMAG + ((*(XP+i) * *(YP+i+1)) + (*(XP+i+1) * *(YP+i)));
}
return ;
}
void mncblas_zdotu_sub_vec(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotu)
{
register unsigned int i = 0 ;
double *XP = (double *)X;
double *YP = (double *)Y;
dcomplexe *temp = (dcomplexe *)dotu;
temp->IMAG = 0;
temp->REEL = 0;
__m128d v11, v12, v21, v22, resm1, resm2;
double2 _v11, _v12, _v21, _v22, resd1, resd2;
for (; ((i < N*2)) ; i += incX + 2){
_v11[0] = _v22[1] = *(XP+i);
_v12[1] = _v21[1] = *(YP+i+1);
_v12[0] = _v21[0] = *(YP+i);
_v11[1] = _v22[0] = *(XP+i+1);
v11 = _mm_load_pd(_v11);
v12 = _mm_load_pd(_v12);
v21 = _mm_load_pd(_v21);
v22 = _mm_load_pd(_v22);
resm1 = _mm_mul_pd(v11, v21);
resm2 = _mm_mul_pd(v12, v22);
_mm_store_pd(resd1, resm1);
_mm_store_pd(resd2, resm2);
temp->REEL = temp->REEL + (resd1[0] - resd1[1]);
temp->IMAG = temp->IMAG + (resd2[1] + resd2[0]);
}
}
void mncblas_zdotu_sub_omp(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotu)
{
double *XP = (double *)X;
double *YP = (double *)Y;
dcomplexe *temp = (dcomplexe *)dotu;
double re = 0; double im = 0;
#pragma omp parallel for schedule(static) reduction(+:re, im)
for (register unsigned int i = 0; i < N*2; i += incX + 2){
re = re + ((*(XP+i) * *(YP+i)) - (*(XP+i+1) * *(YP+i+1)));
im = im + ((*(XP+i) * *(YP+i+1)) + (*(XP+i+1) * *(YP+i)));
}
temp->REEL = re;
temp->IMAG = im;
}
void mncblas_cdotc_sub(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotc)
{
register unsigned int i = 0 ;
float *XP = (float *)X;
float *YP = (float *)Y;
vcomplexe *temp = (vcomplexe *)dotc;
temp->IMAG = 0;
temp->REEL = 0;
for (; ((i < N*2)) ; i += incX + 2){
temp->REEL = temp->REEL + ((*(XP+i) * *(YP+i)) - ((0-*(XP+i+1)) * *(YP+i+1)));
temp->IMAG = temp->IMAG + ((*(XP+i) * *(YP+i+1)) + ((0-*(XP+i+1)) * *(YP+i)));
}
return ;
}
void mncblas_cdotc_sub_omp(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotc)
{
float *XP = (float *)X;
float *YP = (float *)Y;
vcomplexe *temp = (vcomplexe *)dotc;
double re = 0; double im = 0;
#pragma omp parallel for schedule(static) reduction(+:re, im)
for (register unsigned int i = 0 ; i < N*2; i += incX + 2){
re = re + ((*(XP+i) * *(YP+i)) - ((0-*(XP+i+1)) * *(YP+i+1)));
im = im + ((*(XP+i) * *(YP+i+1)) + ((0-*(XP+i+1)) * *(YP+i)));
}
temp->REEL = re;
temp->IMAG = im;
return ;
}
void mncblas_cdotc_sub_vec(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotc)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
float *XP = (float *) X;
float *YP = (float *) Y;
vcomplexe *temp = (vcomplexe *)dotc;
__m128 o1, o2, m, r;
float4 fdot;
for (; ((i < N*2) && (j < N*2)) ; i += incX + 2, j+=incY + 2)
{
o1 = _mm_set_ps((0-*(XP+i+1)), (0-*(XP+i+1)), *(XP+i), *(XP+i));
o2 = _mm_set_ps(*(YP+j), *(YP+j+1), *(YP+j+1), *(YP+j));
m = _mm_mul_ps(o1, o2);
r = _mm_addsub_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 1, 3, 2)));
_mm_store_ps(fdot, r);
temp->REEL = temp->REEL + fdot[0];
temp->IMAG = temp->IMAG + fdot[1];
}
return ;
}
void mncblas_zdotc_sub(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotc)
{
double *XP = (double *)X;
double *YP = (double *)Y;
dcomplexe *temp = (dcomplexe *)dotc;
temp->IMAG = 0;
temp->REEL = 0;
for (register unsigned int i = 0; i < N*2; i += incX + 2){
register double conj = (0-*(XP+i+1));
temp->REEL = temp->REEL + ((*(XP+i) * *(YP+i)) - (conj * *(YP+i+1)));
temp->IMAG = temp->IMAG + ((*(XP+i) * *(YP+i+1)) + (conj * *(YP+i)));
}
return ;
}
void   mncblas_zdotc_sub_omp(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotc)
{
double *XP = (double *)X;
double *YP = (double *)Y;
dcomplexe *temp = (dcomplexe *)dotc;
register double re = 0;
register double im = 0;
#pragma omp parallel for schedule(static) reduction(+:re, im)
for (register unsigned int i = 0; i < N*2; i += incX + 2){
register double conj = (0-*(XP+i+1));
re = re + ((*(XP+i) * *(YP+i)) - (conj * *(YP+i+1)));
im = im + ((*(XP+i) * *(YP+i+1)) + (conj * *(YP+i)));
}
temp->IMAG = im;
temp->REEL = re;
return ;
}
void mncblas_zdotc_sub_vec(const int N, const void *X, const int incX,
const void *Y, const int incY, void *dotc)
{
register unsigned int i = 0 ;
register unsigned int j = 0 ;
double *XP = (double *) X;
double *YP = (double *) Y;
dcomplexe *temp = (dcomplexe *)dotc;
__m128d o11, o12, o21, o22, m1, m2, r;
double2 fdot, to11, to12, to21, to22;
for (; ((i < N*2) && (j < N*2)) ; i += incX + 2, j+=incY + 2)
{
register double conj = (0-*(XP+i+1));
o11 = _mm_set_pd(*(XP+i), *(XP+i));
o12 = _mm_set_pd(conj, conj);
o22 = _mm_set_pd(*(YP+j), *(YP+i+1));
o21 = _mm_set_pd(*(YP+j+1), *(YP+j));
m1 = _mm_mul_pd(o11, o21);
m2 = _mm_mul_pd(o12, o22);
r = _mm_addsub_pd(m1, m2);
_mm_store_pd(fdot, r);
temp->REEL = temp->REEL + fdot[0];
temp->IMAG = temp->IMAG + fdot[1];
}
return ;
}
int main(){
dcomplexe v1[5] = {{1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}, {1.0, 2.0}};
dcomplexe v2[5] = {{3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}, {3.0, 4.0}};
dcomplexe r;
mncblas_zdotc_sub_vec(4, v1, 0, v2, 0, &r);
printf("complexe : %f %f \n", r.REEL, r.IMAG);
}