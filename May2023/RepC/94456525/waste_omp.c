#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
int main(int argc, char *argv[]) {
int ntarget, i,j;
float *xa, x; 
if ( argc != 2 ) {
printf( "Usage: %s ncosmic_rays \n", argv[0] );
exit(0);
} 
sscanf(argv[1], "%i", &ntarget); 
xa = (float *)malloc(sizeof(float)*ntarget); 
memset(xa, 0, sizeof(int)*ntarget); 
srand(time(NULL)); 
#pragma omp parallel for private(x,j) 
for (i=0; i<ntarget; i++) {
for (j=0; j<100; j++) {
x=((float)rand()/(float)(RAND_MAX));
}
xa[i]=x; 
}
return(0);	
}