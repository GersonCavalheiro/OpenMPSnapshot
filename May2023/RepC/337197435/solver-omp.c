#include "heat.h"
#include "omp.h"
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey)
{
#pragma omp parallel for
for (int i=1; i<=sizex-2; i++)
for (int j=1; j<=sizey-2; j++) 
v[ i*sizey+j ] = u[ i*sizey+j ];
}
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
double sum=0.0;
#pragma omp parallel reduction(+:sum)
{
double diff;
int myid = omp_get_thread_num();
int numproc = omp_get_num_threads();
int i_start = lowerb(myid, numproc, sizex);
int i_end = upperb(myid, numproc, sizex);
for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
for (int j=1; j<= sizey-2; j++) {
utmp[i*sizey+j]= 0.25 * ( u[ i*sizey     + (j-1) ]+  
u[ i*sizey     + (j+1) ]+  
u[ (i-1)*sizey + j     ]+  
u[ (i+1)*sizey + j     ]); 
diff = utmp[i*sizey+j] - u[i*sizey + j];
sum += diff * diff; 
}
}
}
return sum;
}
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
double unew, diff, sum=0.0;
#pragma omp parallel
{
int howmany=omp_get_num_threads();
#pragma omp for ordered(2) private(unew, diff) reduction(+:sum)
for (int blockid_row = 0; blockid_row < howmany; ++blockid_row) {
for (int blockid_col = 0; blockid_col < howmany; ++blockid_col) {
int i_start = lowerb(blockid_row, howmany, sizex);
int i_end = upperb(blockid_row, howmany, sizex);
int j_start = lowerb(blockid_col, howmany, sizey);
int j_end = upperb(blockid_col, howmany, sizey);
#pragma omp ordered depend (sink: blockid_row-1, blockid_col)
for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
for (int j=max(1, j_start); j<= min(sizey-2, j_end); j++) {
unew= 0.25 * ( u[ i*sizey   + (j-1) ]+  
u[ i*sizey   + (j+1) ]+  
u[ (i-1)*sizey       + j     ]+  
u[ (i+1)*sizey       + j     ]); 
diff = unew - u[i*sizey+ j];
sum += diff * diff; 
u[i*sizey+j]=unew;
}
}
#pragma omp ordered depend(source)
}
}
}
return sum;
}
