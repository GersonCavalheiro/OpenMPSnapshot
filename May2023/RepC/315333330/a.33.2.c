#include <stdio.h>
#include <stdlib.h>
float
read_next ()
{
float *tmp;
float return_val;
#pragma omp single copyprivate(tmp)
{
tmp = (float *) malloc (sizeof (float));
}				
#pragma omp master
{
scanf ("%f", tmp);
}
#pragma omp barrier
return_val = *tmp;
#pragma omp barrier
#pragma omp single nowait
{
free (tmp);
}
return return_val;
}
