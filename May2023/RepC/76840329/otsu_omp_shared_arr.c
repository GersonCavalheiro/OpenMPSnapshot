#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include "mypgm_omp.h"
void otsu_th( )
{
struct timeval TimeValue_Start;
struct timezone TimeZone_Start;
struct timeval TimeValue_Final;
struct timezone TimeZone_Final;
long time_start, time_end;
double time_overhead;
int hist[GRAYLEVEL];
double prob[GRAYLEVEL], omega[GRAYLEVEL]; 
double myu[GRAYLEVEL];   
double max_sigma, sigma[GRAYLEVEL]; 
int i, x, y; 
int threshold; 
printf("Otsu's binarization process starts now.\n");
int* h1;
#pragma omp parallel
{
const int nthreads = omp_get_num_threads();
const int ithread = omp_get_thread_num();
printf("%d",nthreads);
#pragma omp single 
{
h1 = new int[GRAYLEVEL*nthreads];
for(int i=0; i<(GRAYLEVEL*nthreads); i++) h1[i] = 0;
}
#pragma omp for schedule(dynamic,20)
for (int n=0 ; n<y_size1;n++ )
{
for (int m=0; m<x_size1; m++){
h1[ithread*GRAYLEVEL+image1[n][m]]++;
}
}
#pragma omp for schedule(dynamic,20)
for(int i=0; i<GRAYLEVEL; i++) {
for(int t=0; t<nthreads; t++) {
hist[i] += h1[GRAYLEVEL*t + i];
}
}
#pragma omp for schedule(dynamic,20)  
for ( i = 0; i < GRAYLEVEL; i ++ ) {
prob[i] = (double)hist[i] / (x_size1 * y_size1);
}
#pragma omp single
{  omega[0] = prob[0];
myu[0] = 0.0;       
for (i = 1; i < GRAYLEVEL; i++) {
omega[i] = omega[i-1] + prob[i];
myu[i] = myu[i-1] + i*prob[i];
}}
threshold = 0;
max_sigma = 0.0;
#pragma omp for schedule(dynamic,20) reduction(max:max_sigma)
for (i = 0; i < GRAYLEVEL-1; i++) {
if (omega[i] != 0.0 && omega[i] != 1.0)
sigma[i] = pow(myu[GRAYLEVEL-1]*omega[i] - myu[i], 2) / (omega[i]*(1.0 - omega[i]));
else
sigma[i] = 0.0;
if (sigma[i] > max_sigma) {
max_sigma = sigma[i];
threshold = i;
}
}}
x_size2 = x_size1;
y_size2 = y_size1;
#pragma omp for collapse(2) private(x,y) schedule(dynamic,5000)
for (y = 0; y < y_size2; y++)
for (x = 0; x < x_size2; x++)
if (image1[y][x] > threshold)
image2[y][x] = MAX_BRIGHTNESS;
else
image2[y][x] = 0;
}
main( )
{ 
load_image_data( ); 
clock_t begin = clock();
otsu_th( );         
clock_t end = clock();
double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
printf("TIME TAKEN: %f\n",time_spent);
save_image_data( ); 
return 0;
}
