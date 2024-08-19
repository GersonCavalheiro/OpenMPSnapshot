#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "matrix.h"
#include "config.h"
#include "complex.h"
#include "error.h"
unsigned long quantum_memman(long change)
{
static long mem = 0, max = 0;
mem += change;
if(mem > max)
max = mem;
return mem;
}
void quantum_matrix_scanf(quantum_matrix *m){
int i, j, z=0;
while ((1 << z++) < m->rows);
z--;
for(i=0; i<m->rows; i++) 
{
for(j=0; j<m->cols; j++){
}
printf("\n");
}
printf("\n");
}
quantum_matrix quantum_new_matrix(int cols, int rows) 
{
quantum_matrix m;
m.rows = rows;
m.cols = cols;
m.t = calloc(cols * rows, sizeof(COMPLEX_FLOAT));
#if (DEBUG_MEM)
printf("allocating %i bytes of memory for %ix%i matrix at 0x%X\n",
sizeof(COMPLEX_FLOAT) * cols * rows, cols, rows, (int) m.t);
#endif  
if(!m.t)
quantum_error(QUANTUM_ENOMEM);
quantum_memman(sizeof(COMPLEX_FLOAT) * cols * rows);
return m;
}
void
quantum_delete_matrix(quantum_matrix *m)
{
#if (DEBUG_MEM)	
printf("freeing %i bytes of memory for %ix%i matrix at 0x%X\n",
sizeof(COMPLEX_FLOAT) * m->cols * m->rows, m->cols, m->rows,
(int) m->t);	
#endif  
free(m->t);
quantum_memman(-sizeof(COMPLEX_FLOAT) * m->cols * m->rows);
m->t=0;
}
void quantum_print_matrix(quantum_matrix m) 
{
int i, j, z=0;
while ((1 << z++) < m.rows);
z--;
#pragma omp parallel for
{
#pragma omp parallel for
for(j=0; j<m.cols; j++)
printf("%g %+gi ", quantum_real(M(m, j, i)), quantum_imag(M(m, j, i)));
printf("\n");
}
printf("\n");
}
quantum_matrix quantum_mmult(quantum_matrix A, quantum_matrix B)
{
int i, j, k;
quantum_matrix C;
if(A.cols != B.rows)
quantum_error(QUANTUM_EMSIZE);
C = quantum_new_matrix(B.cols, A.rows);
#pragma omp parallel for
for(i=0; i<B.cols; i++)
{
for(j=0; j<A.rows; j++)
{
for(k=0; k<B.rows; k++)
M(C, i, j) += M(A, k, j) * M(B, i, k);
}
}
return C;
}
