





#include "headers.h"
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif



int
hypre_CSRMatrixMatvec( double           alpha,
hypre_CSRMatrix *A,
hypre_Vector    *x,
double           beta,
hypre_Vector    *y     )
{
double     *A_data   = hypre_CSRMatrixData(A);
int        *A_i      = hypre_CSRMatrixI(A);
int        *A_j      = hypre_CSRMatrixJ(A);
int         num_rows = hypre_CSRMatrixNumRows(A);
int         num_cols = hypre_CSRMatrixNumCols(A);

int        *A_rownnz = hypre_CSRMatrixRownnz(A);
int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

double     *x_data = hypre_VectorData(x);
double     *y_data = hypre_VectorData(y);
int         x_size = hypre_VectorSize(x);
int         y_size = hypre_VectorSize(y);
int         num_vectors = hypre_VectorNumVectors(x);
int         idxstride_y = hypre_VectorIndexStride(y);
int         vecstride_y = hypre_VectorVectorStride(y);
int         idxstride_x = hypre_VectorIndexStride(x);
int         vecstride_x = hypre_VectorVectorStride(x);

double      temp, tempx;

int         i, j, jj;

int         m;

double     xpar=0.7;

int         ierr = 0;




hypre_assert( num_vectors == hypre_VectorNumVectors(y) );

if (num_cols != x_size)
ierr = 1;

if (num_rows != y_size)
ierr = 2;

if (num_cols != x_size && num_rows != y_size)
ierr = 3;



if (alpha == 0.0)
{
for (i = 0; i < num_rows*num_vectors; i++)
y_data[i] *= beta;

return ierr;
}



temp = beta / alpha;

if (temp != 1.0)
{
if (temp == 0.0)
{
for (i = 0; i < num_rows*num_vectors; i++)
y_data[i] = 0.0;
}
else
{
for (i = 0; i < num_rows*num_vectors; i++)
y_data[i] *= temp;
}
}





if (num_rownnz < xpar*(num_rows))
{
for (i = 0; i < num_rownnz; i++)
{
m = A_rownnz[i];


if ( num_vectors==1 )
{
tempx = y_data[m];
for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
tempx +=  A_data[jj] * x_data[A_j[jj]];
y_data[m] = tempx;
}
else
for ( j=0; j<num_vectors; ++j )
{
tempx = y_data[ j*vecstride_y + m*idxstride_y ];
for (jj = A_i[m]; jj < A_i[m+1]; jj++) 
tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
y_data[ j*vecstride_y + m*idxstride_y] = tempx;
}
}

}
else
{
#ifdef _OPENMP
#pragma omp parallel for private(i,jj,temp) schedule(static)
#endif
for (i = 0; i < num_rows; i++)
{
if ( num_vectors==1 )
{
temp = y_data[i];
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
temp += A_data[jj] * x_data[A_j[jj]];
y_data[i] = temp;
}
else
for ( j=0; j<num_vectors; ++j )
{
temp = y_data[ j*vecstride_y + i*idxstride_y ];
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
{
temp += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
}
y_data[ j*vecstride_y + i*idxstride_y ] = temp;
}
}
}




if (alpha != 1.0)
{
for (i = 0; i < num_rows*num_vectors; i++)
y_data[i] *= alpha;
}

return ierr;
}



int
hypre_CSRMatrixMatvecT( double           alpha,
hypre_CSRMatrix *A,
hypre_Vector    *x,
double           beta,
hypre_Vector    *y     )
{
double     *A_data    = hypre_CSRMatrixData(A);
int        *A_i       = hypre_CSRMatrixI(A);
int        *A_j       = hypre_CSRMatrixJ(A);
int         num_rows  = hypre_CSRMatrixNumRows(A);
int         num_cols  = hypre_CSRMatrixNumCols(A);

double     *x_data = hypre_VectorData(x);
double     *y_data = hypre_VectorData(y);
int         x_size = hypre_VectorSize(x);
int         y_size = hypre_VectorSize(y);
int         num_vectors = hypre_VectorNumVectors(x);
int         idxstride_y = hypre_VectorIndexStride(y);
int         vecstride_y = hypre_VectorVectorStride(y);
int         idxstride_x = hypre_VectorIndexStride(x);
int         vecstride_x = hypre_VectorVectorStride(x);

double      temp;

int         i, i1, j, jv, jj, ns, ne, size, rest;
int         num_threads;

int         ierr  = 0;



hypre_assert( num_vectors == hypre_VectorNumVectors(y) );

if (num_rows != x_size)
ierr = 1;

if (num_cols != y_size)
ierr = 2;

if (num_rows != x_size && num_cols != y_size)
ierr = 3;


if (alpha == 0.0)
{

for (i = 0; i < num_cols*num_vectors; i++)
y_data[i] *= beta;

return ierr;
}



temp = beta / alpha;

if (temp != 1.0)
{
if (temp == 0.0)
{

for (i = 0; i < num_cols*num_vectors; i++)
y_data[i] = 0.0;
}
else
{

for (i = 0; i < num_cols*num_vectors; i++)
y_data[i] *= temp;
}
}


num_threads = hypre_NumThreads();
if (num_threads > 1)
{


for (i1 = 0; i1 < num_threads; i1++)
{
size = num_cols/num_threads;
rest = num_cols - size*num_threads;
if (i1 < rest)
{
ns = i1*size+i1-1;
ne = (i1+1)*size+i1+1;
}
else
{
ns = i1*size+rest-1;
ne = (i1+1)*size+rest;
}
if ( num_vectors==1 )
{
for (i = 0; i < num_rows; i++)
{
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
{
j = A_j[jj];
if (j > ns && j < ne)
y_data[j] += A_data[jj] * x_data[i];
}
}
}
else
{
for (i = 0; i < num_rows; i++)
{
for ( jv=0; jv<num_vectors; ++jv )
{
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
{
j = A_j[jj];
if (j > ns && j < ne)
y_data[ j*idxstride_y + jv*vecstride_y ] +=
A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x];
}
}
}
}

}
}
else 
{
for (i = 0; i < num_rows; i++)
{
if ( num_vectors==1 )
{
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
{
j = A_j[jj];
y_data[j] += A_data[jj] * x_data[i];
}
}
else
{
for ( jv=0; jv<num_vectors; ++jv )
{
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
{
j = A_j[jj];
y_data[ j*idxstride_y + jv*vecstride_y ] +=
A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
}
}
}
}
}


if (alpha != 1.0)
{

for (i = 0; i < num_cols*num_vectors; i++)
y_data[i] *= alpha;
}

return ierr;
}



int
hypre_CSRMatrixMatvec_FF( double           alpha,
hypre_CSRMatrix *A,
hypre_Vector    *x,
double           beta,
hypre_Vector    *y,
int             *CF_marker_x,
int             *CF_marker_y,
int fpt )
{
double     *A_data   = hypre_CSRMatrixData(A);
int        *A_i      = hypre_CSRMatrixI(A);
int        *A_j      = hypre_CSRMatrixJ(A);
int         num_rows = hypre_CSRMatrixNumRows(A);
int         num_cols = hypre_CSRMatrixNumCols(A);

double     *x_data = hypre_VectorData(x);
double     *y_data = hypre_VectorData(y);
int         x_size = hypre_VectorSize(x);
int         y_size = hypre_VectorSize(y);

double      temp;

int         i, jj;

int         ierr = 0;




if (num_cols != x_size)
ierr = 1;

if (num_rows != y_size)
ierr = 2;

if (num_cols != x_size && num_rows != y_size)
ierr = 3;



if (alpha == 0.0)
{

for (i = 0; i < num_rows; i++)
if (CF_marker_x[i] == fpt) y_data[i] *= beta;

return ierr;
}



temp = beta / alpha;

if (temp != 1.0)
{
if (temp == 0.0)
{

for (i = 0; i < num_rows; i++)
if (CF_marker_x[i] == fpt) y_data[i] = 0.0;
}
else
{

for (i = 0; i < num_rows; i++)
if (CF_marker_x[i] == fpt) y_data[i] *= temp;
}
}





for (i = 0; i < num_rows; i++)
{
if (CF_marker_x[i] == fpt)
{
temp = y_data[i];
for (jj = A_i[i]; jj < A_i[i+1]; jj++)
if (CF_marker_y[A_j[jj]] == fpt) temp += A_data[jj] * x_data[A_j[jj]];
y_data[i] = temp;
}
}




if (alpha != 1.0)
{

for (i = 0; i < num_rows; i++)
if (CF_marker_x[i] == fpt) y_data[i] *= alpha;
}

return ierr;
}
