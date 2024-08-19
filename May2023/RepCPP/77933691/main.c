#include <stdio.h>
#include <stdlib.h>
#include "mandel.h"
#include "writepng.h"
#include <omp.h>

int
main(int argc, char *argv[]) {

int   width, height;
int	  max_iter;
int   *image;

width    = 2601;
height   = 2601;
max_iter = 400;

if ( argc == 2 ) width = height = atoi(argv[1]);

image = (int *)malloc( width * height * sizeof(int));
if ( image == NULL ) {
fprintf(stderr, "memory allocation failed!\n");
return(1);
}

double ts, te;
ts = omp_get_wtime();

#pragma omp parallel
{
mandel(width, height, image, max_iter);
}

te = omp_get_wtime() - ts;
printf("%lf\n", te);


return(0);
}
