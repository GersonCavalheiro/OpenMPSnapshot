#include "heat.h"
#define NB 8
#define min(a,b) ( ((a) < (b)) ? (a) : (b) )
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
double diff, sum=0.0;
int nbx, bx, nby, by;
nbx = omp_get_max_threads();
bx = sizex/nbx + ((sizex%nbx) ? 1 : 0);
nby = 1;
by = sizey/nby;
#pragma omp parallel for reduction(+:sum) private(diff)
for (int ii=0; ii<nbx; ii++) {
for (int jj=0; jj<nby; jj++)  {
for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) {
for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
utmp[i*sizey+j]= 0.25 * (u[ i*sizey	    + (j-1) ]+  
u[ i*sizey     + (j+1) ]+  
u[ (i-1)*sizey + j     ]+  
u[ (i+1)*sizey + j     ]); 
diff = utmp[i*sizey+j] - u[i*sizey + j];
sum += diff * diff; 
}
}
}
}
return sum;
}
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
double unew, diff, sum=0.0;
int nbx, bx, nby, by;
int lsw;
/
#pragma omp parallel for reduction(+:sum) private(diff, lsw)
for (int ii=0; ii<nbx; ii++) {
lsw = ii%2;
for (int jj=lsw; jj<nby; jj=jj+2) {
for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) {
for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
unew= 0.25 * (	u[ i*sizey     + (j-1) ]+  
u[ i*sizey     + (j+1) ]+  
u[ (i-1)*sizey	+ j     ]+  
u[ (i+1)*sizey	+ j     ]); 
diff = unew - u[i*sizey+ j];
sum += diff * diff;
u[i*sizey+j]=unew;
}
}
}
}
#pragma omp parallel for reduction(+:sum) private(diff, lsw)
for (int ii=0; ii<nbx; ii++) {
lsw = (ii+1)%2;
for (int jj=lsw; jj<nby; jj=jj+2) {
for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) {
for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
unew= 0.25 * (	u[ i*sizey     + (j-1) ]+  
u[ i*sizey     + (j+1) ]+  
u[ (i-1)*sizey	+ j     ]+  
u[ (i+1)*sizey	+ j     ]); 
diff = unew - u[i*sizey+ j];
sum += diff * diff; 
u[i*sizey+j]=unew;
}
}
}
}
return sum;
}
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
double unew, diff, sum=0.0;
int nbx, bx, nby, by;
/
#pragma omp parallel for reduction(+:sum) private(diff)
for (int ii=0; ii<nbx; ii++)
for (int jj=0; jj<nby; jj++) 
for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
unew= 0.25 * (	u[ i*sizey     + (j-1) ]+  
u[ i*sizey     + (j+1) ]+  
u[ (i-1)*sizey	+ j     ]+  
u[ (i+1)*sizey	+ j     ]); 
diff = unew - u[i*sizey+ j];
sum += diff * diff; 
u[i*sizey+j]=unew;
}
return sum;
}
