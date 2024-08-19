#include <omp.h>
#include <stdbool.h>
#include "mp_comp_MM_det.h"
#include "compute_det4x4real.h"
void mp_comp_MM_det( double *Mdet_out, bool *MdetMsk_out, double *M_in,
double *MdetThr_in, int *idx_in, int *numel_in )
{
int m = 16;
#pragma omp parallel for 
for (int i=0; i<numel_in[0]; ++i) 
{
compute_det4x4real( &Mdet_out[idx_in[i]],
&M_in[idx_in[i]*m+0] , &M_in[idx_in[i]*m+1] , &M_in[idx_in[i]*m+2]  , &M_in[idx_in[i]*m+3],
&M_in[idx_in[i]*m+4] , &M_in[idx_in[i]*m+5] , &M_in[idx_in[i]*m+6]  , &M_in[idx_in[i]*m+7],
&M_in[idx_in[i]*m+8] , &M_in[idx_in[i]*m+9] , &M_in[idx_in[i]*m+10] , &M_in[idx_in[i]*m+11],
&M_in[idx_in[i]*m+12], &M_in[idx_in[i]*m+13], &M_in[idx_in[i]*m+14] , &M_in[idx_in[i]*m+15] );
if (Mdet_out[idx_in[i]] < *MdetThr_in)
{MdetMsk_out[idx_in[i]]=0;}
else
{MdetMsk_out[idx_in[i]]=1;}
}
}