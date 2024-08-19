

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


struct volume { 
float* x; 
float* y;
float* z;
float* mass;
};



int place_uniformly(int sx, int ex, int sy, int ey, int sz, int ez, struct volume* v) {
int counter = 0;
int number_of_phaseballs = (ex-sx+1)*(ey-sy+1)*(ez-sz+1);

v->x = (float*) malloc(number_of_phaseballs * sizeof(float));
v->y = (float*) malloc(number_of_phaseballs * sizeof(float));
v->z = (float*) malloc(number_of_phaseballs * sizeof(float));
v->mass = (float*) malloc(number_of_phaseballs * sizeof(float));
for(int i=sx; i<=ex; i++) {
for(int j=sy; j<=ey; j++) {
for(int k=sz; k<=ez; k++) {
v->x[counter] = i; 
v->y[counter] = j; 
v->z[counter] = k;
v->mass[counter] = fabs(v->x[counter])+fabs(v->y[counter])+fabs(v->z[counter]);
counter++;
}
}
}
return number_of_phaseballs;
}


void post_process(struct volume* v, float* cx, float* cy,int length
) {
double mass_sum=0.0;
double wx=0.0;
double wy=0.0;
#pragma omp parallel for reduction(+:mass_sum,wx,wy)
for(int i=0; i<length; i++) {
mass_sum += v->mass[i];
wx += v->x[i] * v->mass[i];
wy +=v->y[i] * v->mass[i];
}
*cx = wx/mass_sum; 
*cy = wy/mass_sum;

return;
}

int main(int argc, char** argv) {
struct volume v;
int lengtharray = place_uniformly(-1000,1000,-100,100,-100,100,&v); 


float cx,cy;
struct timespec start_time;
struct timespec end_time;
clock_gettime(CLOCK_MONOTONIC,&start_time);
post_process(&v, &cx, &cy,lengtharray); 
clock_gettime(CLOCK_MONOTONIC,&end_time);
long msec = (end_time.tv_sec - start_time.tv_sec)*1000 + (end_time.tv_nsec - start_time.tv_nsec)/1000000;

printf("Centroid at: %f,%f\n",cx,cy);
printf("found in %dms\n",msec);

return EXIT_SUCCESS;
}
