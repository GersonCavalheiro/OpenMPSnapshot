#include<stdint.h>
#include<math.h>
#include<omp.h>
#include<ndlib.h>
void zoomOutData( uint32_t * olddata, uint32_t * newdata, int * dims, int factor )
{
int i,j,k;
int zdim = dims[0];
int ydim = dims[1];
int xdim = dims[2];
int oldindex,newindex;
int power = pow(2,factor);
for ( i=0; i<zdim; i++ )
for ( j=0; j<ydim; j++ )
for ( k=0; k<xdim; k++ )
{
newindex = (i*xdim*ydim)+(j*xdim)+(k);
oldindex = ( i*(xdim*power)*(ydim*power) ) + ( (j*power)*(xdim*power) ) + ( k*power );
newdata[newindex] = olddata[oldindex];
}
}
void zoomOutDataOMP( uint32_t * olddata, uint32_t * newdata, int * dims, int factor )
{
int i,j,k;
int zdim = dims[0];
int ydim = dims[1];
int xdim = dims[2];
int oldindex,newindex;
int power = pow(2,factor);
#pragma omp parallel num_threads(omp_get_max_threads())
{
#pragma omp for private(i,j,k) schedule(dynamic)
for ( i=0; i<zdim; i++ )
for ( j=0; j<ydim; j++ )
for ( k=0; k<xdim; k++ )
{
newindex = (i*xdim*ydim)+(j*xdim)+(k);
oldindex = ( i*(xdim*power)*(ydim*power) ) + ( (j*power)*(xdim*power) ) + ( k*power );
newdata[newindex] = olddata[oldindex];
}
}
}
void zoomInData( uint32_t * olddata, uint32_t * newdata, int * dims, int factor )
{
int i,j,k;
int zdim = dims[0];
int ydim = dims[1];
int xdim = dims[2];
int oldindex,newindex;
int power = pow(2,factor);
for ( i=0; i<zdim; i++ )
for ( j=0; j<ydim; j++ )
for ( k=0; k<xdim; k++ )
{
newindex = (i*xdim*ydim)+(j*xdim)+(k);
oldindex = ( i*(xdim/power)*(ydim/power) ) + ( (j/power)*(xdim/power) ) + ( k/power );
newdata[newindex] = olddata[oldindex];
}
}
void zoomInDataOMP16( uint16_t * olddata, uint16_t * newdata, int * dims, int factor )
{
int i,j,k;
int zdim = dims[0];
int ydim = dims[1];
int xdim = dims[2];
int oldindex, newindex;
int power = pow(2,factor);
#pragma omp parallel num_threads(omp_get_max_threads())
{
#pragma omp for private(i,j,k) schedule(dynamic)
for ( i=0; i<zdim; i++ )
for ( j=0; j<ydim; j++ )
for ( k=0; k<xdim; k++ )
{
newindex = (i*xdim*ydim)+(j*xdim)+(k);
oldindex = ( i*(xdim/power)*(ydim/power) ) + ( (j/power)*(xdim/power) ) + ( k/power );
newdata[newindex] = olddata[oldindex];
}
}
}
void zoomInDataOMP32( uint32_t * olddata, uint32_t * newdata, int * dims, int factor )
{
int i,j,k;
int zdim = dims[0];
int ydim = dims[1];
int xdim = dims[2];
int oldindex, newindex;
int power = pow(2,factor);
#pragma omp parallel num_threads(omp_get_max_threads())
{
#pragma omp for private(i,j,k) schedule(dynamic)
for ( i=0; i<zdim; i++ )
for ( j=0; j<ydim; j++ )
for ( k=0; k<xdim; k++ )
{
newindex = (i*xdim*ydim)+(j*xdim)+(k);
oldindex = ( i*(xdim/power)*(ydim/power) ) + ( (j/power)*(xdim/power) ) + ( k/power );
newdata[newindex] = olddata[oldindex];
}
}
}
