#include<stdint.h>
#include<omp.h>
#include<ndlib.h>
void recolorCubeOMP32 ( uint32_t * cutout, int xdim, int ydim, uint32_t * imagemap, uint32_t * rgbColor)
{
int i,j;
#pragma omp parallel num_threads( omp_get_max_threads() )
{
#pragma omp for private(i,j) schedule(dynamic)
for ( i=0; i<xdim; i++)
for ( j=0; j<ydim; j++)
if ( cutout [(i*ydim)+j] != 0 )
imagemap [(i*ydim)+j] = rgbColor[ cutout [(i*ydim)+j] % 217 ];
}
}
void recolorCubeOMP64 ( uint64_t * cutout, int xdim, int ydim, uint64_t * imagemap, uint64_t * rgbColor)
{
int i,j;
#pragma omp parallel num_threads( omp_get_max_threads() )
{
#pragma omp for private(i,j) schedule(dynamic)
for ( i=0; i<xdim; i++)
for ( j=0; j<ydim; j++)
if ( cutout [(i*ydim)+j] != 0 )
imagemap [(i*ydim)+j] = rgbColor[ cutout [(i*ydim)+j] % 217 ];
}
}
