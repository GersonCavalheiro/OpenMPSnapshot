#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
int main(int argc, char *argv[]) {
int ntarget, i;
float *xa, *ya; 
float x, y, sampling, exposure;
if ( argc != 2 ) {
printf( "Usage: %s ncosmic_rays \n", argv[0] );
exit(0);
} 
sscanf(argv[1], "%i", &ntarget); 
xa = (float *)malloc(sizeof(float)*ntarget); 
ya = (float *)malloc(sizeof(float)*ntarget); 
memset(xa, 0, sizeof(int)*ntarget); 
memset(ya, 0, sizeof(int)*ntarget);
srand(time(NULL)); 
#pragma omp parallel for private(x,y,i) 
for (i=0; i<ntarget; i++) {
x=((float)rand()/(float)(RAND_MAX)) * 360.;
y=((float)rand()/(float)(RAND_MAX));
y = asin(2.*y-1.)*180./M_PI;	
xa[i]=x; 
ya[i]=y;
}
return(0);	
}