








#define ATTEMPTS 12
#define MIN_SCALE_FACTOR 0.125
#define MAX_SCALE_FACTOR 4.0

#include <stdlib.h>                  
#include <math.h>                  
#include "./embedded_fehlberg_7_8.c"

int solver(
fp **y,
fp *x,
int xmax,
fp *params,
fp *com,
double *timecopyin,
double *timecopykernel,
double *timecopyout)
{


fp err_exponent;
int error;
int outside;
fp h;
fp h_init;
fp tolerance;
int xmin;

fp scale_min;
fp scale_fina;
fp* err= (fp *) malloc(EQUATIONS* sizeof(fp));
fp* scale= (fp *) malloc(EQUATIONS* sizeof(fp));
fp* yy= (fp *) malloc(EQUATIONS* sizeof(fp));

int i, j, k;


err_exponent = 1.0 / 7.0;
h_init = 1;
h = h_init;
xmin = 0;
tolerance = 10 / (fp)(xmax-xmin);

x[0] = 0;


if (xmax < xmin || h <= 0.0){
return -2;
}

if (xmax == xmin){
return 0; 
}

if (h > (xmax - xmin) ) { 
h = (fp)xmax - (fp)xmin; 
}


#ifdef DEBUG
printf("Time Steps: ");
fflush(0);
#endif

#pragma omp target enter data map(alloc: params[0:PARAMETERS], com[0:3])
for(k=1; k<=xmax; k++) {                      

x[k] = k-1;
h = h_init;


scale_fina = 1.0;


for (j = 0; j < ATTEMPTS; j++) {


error = 0;
outside = 0;
scale_min = MAX_SCALE_FACTOR;


embedded_fehlberg_7_8(  x[k],
h,
y[k-1],
y[k],
params,
com,
err,
timecopyin,
timecopykernel,
timecopyout);


for(i=0; i<EQUATIONS; i++){
if(err[i] > 0){
error = 1;
}
}
if (error != 1) {
scale_fina = MAX_SCALE_FACTOR; 
break;
}


for(i=0; i<EQUATIONS; i++){
if(y[k-1][i] == 0.0){
yy[i] = tolerance;
}
else{
yy[i] = fabs(y[k-1][i]);
}
scale[i] = 0.8 * pow( tolerance * yy[i] / err[i] , err_exponent );
if(scale[i]<scale_min){
scale_min = scale[i];
}
}

#define max(x,y) ( (x) < (y) ? (y) : (x) )
#define min(x,y) ( (x) < (y) ? (x) : (y) )
scale_fina = min( max(scale_min,MIN_SCALE_FACTOR), MAX_SCALE_FACTOR);


for(i=0; i<EQUATIONS; i++){
if ( err[i] > ( tolerance * yy[i] ) ){
outside = 1;
}
}
if (outside == 0){
break;
}


h = h * scale_fina;

if (h >= 0.9) {
h = 0.9;
}

if ( x[k] + h > (fp)xmax ){
h = (fp)xmax - x[k];
}

else if ( x[k] + h + 0.5 * h > (fp)xmax ){
h = 0.5 * h;
}

}


x[k] = x[k] + h;


if ( j >= ATTEMPTS ) {
#pragma omp target exit data map(release: params[0:PARAMETERS], com[0:3])
return -1; 
}

#ifdef DEBUG
printf("%d ", k);
fflush(0);
#endif

}
#pragma omp target exit data map(release: params[0:PARAMETERS], com[0:3])

#ifdef DEBUG
printf("\n");
fflush(0);
#endif


free(err);
free(scale);
free(yy);

return 0;

} 
