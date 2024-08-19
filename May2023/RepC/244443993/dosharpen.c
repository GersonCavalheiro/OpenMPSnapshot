#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sharpen.h"
#include "utilities.h"
double **dosharpen(char *infile, int nx, int ny)
{
int d = 8;
double norm = (2 * d - 1) * (2 * d - 1);
double scale = 2.0;
int xpix, ypix, pixcount;
int i, j, k, l;
double tstart, tstop, time;
int **fuzzy = int2Dmalloc(nx, ny);                              
double **fuzzyPadded = double2Dmalloc(nx + 2 * d, ny + 2 * d);  
double **convolutionPartial = double2Dmalloc(nx, ny);           
double **convolution = double2Dmalloc(nx, ny);                  
double **sharp = double2Dmalloc(nx, ny);                        
double **sharpCropped = double2Dmalloc(nx - 2 * d, ny - 2 * d); 
char outfile[256];
strcpy(outfile, infile);
*(strchr(outfile, '.')) = '\0';
strcat(outfile, "_sharpened.pgm");
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
fuzzy[i][j] = 0;
sharp[i][j] = 0.0;
}
}
fflush(stdout);
pgmread(infile, &fuzzy[0][0], nx, ny, &xpix, &ypix);
fflush(stdout);
if (xpix == 0 || ypix == 0 || nx != xpix || ny != ypix)
{
printf("Error reading %s\n", infile);
fflush(stdout);
exit(-1);
}
for (i = 0; i < nx + 2 * d; i++)
{
for (j = 0; j < ny + 2 * d; j++)
{
fuzzyPadded[i][j] = 0.0;
}
}
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
fuzzyPadded[i + d][j + d] = fuzzy[i][j];
}
}
fflush(stdout);
tstart = wtime();
pixcount = 0;
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
for (k = -d; k <= d; k++)
{
for (l = -d; l <= d; l++)
{
convolution[i][j] = convolution[i][j] + filter(d, k, l) * fuzzyPadded[i + d + k][j + d + l];
}
}
pixcount += 1;
}
}
tstop = wtime();
time = tstop - tstart;
fflush(stdout);
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
sharp[i][j] = fuzzyPadded[i + d][j + d] - scale / norm * convolution[i][j];
}
}
for (i = d; i < nx - d; i++)
{
for (j = d; j < ny - d; j++)
{
sharpCropped[i - d][j - d] = sharp[i][j];
}
}
pgmwrite(outfile, &sharpCropped[0][0], nx - 2 * d, ny - 2 * d);
fflush(stdout);
free(fuzzy);
free(fuzzyPadded);
free(convolutionPartial);
free(convolution);
free(sharpCropped);
return sharp;
}
double **makeFilterMatrix(int d)
{
double **matrix = (double **)malloc((2 * d + 1) * sizeof(double *));
for (int i = 0; i <= 2 * d; i++)
{
matrix[i] = (double *)malloc((2 * d + 1) * sizeof(double));
for (int j = -d; j <= d; j++)
matrix[i][j + d] = filter(d, i - d, j);
}
return matrix;
}
double **dosharpenParallel(char *infile, int nx, int ny)
{
int d = 8;
double norm = (2 * d - 1) * (2 * d - 1);
double scale = 2.0;
int xpix, ypix, pixcount;
int i, j, k, l;
double tstart, tstop, time;
int **fuzzy = int2Dmalloc(nx, ny);                              
double **fuzzyPadded = double2Dmalloc(nx + 2 * d, ny + 2 * d);  
double **convolutionPartial = double2Dmalloc(nx, ny);           
double **convolution = double2Dmalloc(nx, ny);                  
double **sharp = double2Dmalloc(nx, ny);                        
double **sharpCropped = double2Dmalloc(nx - 2 * d, ny - 2 * d); 
char outfile[256];
strcpy(outfile, infile);
*(strchr(outfile, '.')) = '\0';
strcat(outfile, "_sharpened.pgm");
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
fuzzy[i][j] = 0;
sharp[i][j] = 0.0;
}
}
fflush(stdout);
pgmread(infile, &fuzzy[0][0], nx, ny, &xpix, &ypix);
fflush(stdout);
if (xpix == 0 || ypix == 0 || nx != xpix || ny != ypix)
{
printf("Error reading %s\n", infile);
fflush(stdout);
exit(-1);
}
for (i = 0; i < nx + 2 * d; i++)
{
for (j = 0; j < ny + 2 * d; j++)
{
fuzzyPadded[i][j] = 0.0;
}
}
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
fuzzyPadded[i + d][j + d] = fuzzy[i][j];
}
}
fflush(stdout);
tstart = wtime();
pixcount = 0;
double **filterMatrix = makeFilterMatrix(d);
#pragma omp parallel for collapse(2) default(none) private(i, j, k, l) firstprivate(nx, ny) shared(convolution, fuzzyPadded, filterMatrix, d) schedule(static, 1)
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
for (k = -d; k <= d; k++)
{
for (l = -d; l <= d; l++)
{
convolution[i][j] = convolution[i][j] + filterMatrix[k + d][l + d] * fuzzyPadded[i + d + k][j + d + l];
}
}
}
}
tstop = wtime();
time = tstop - tstart;
fflush(stdout);
double c = scale / norm;
#pragma omp parallel for collapse(2) default(none) private(i, j) shared(d, nx, ny, sharp, fuzzyPadded, c, convolution) schedule(static, 1)
for (i = 0; i < nx; i++)
{
for (j = 0; j < ny; j++)
{
sharp[i][j] = fuzzyPadded[i + d][j + d] - c * convolution[i][j];
}
}
for (i = d; i < nx - d; i++)
{
for (j = d; j < ny - d; j++)
{
sharpCropped[i - d][j - d] = sharp[i][j];
}
}
pgmwrite(outfile, &sharpCropped[0][0], nx - 2 * d, ny - 2 * d);
fflush(stdout);
free(fuzzy);
free(fuzzyPadded);
free(convolutionPartial);
free(convolution);
free(sharpCropped);
free(filterMatrix);
return sharp;
}
int **int2Dmalloc(int nx, int ny)
{
int i;
int **idata;
idata = (int **)malloc(nx * sizeof(int *) + nx * ny * sizeof(int));
idata[0] = (int *)(idata + nx);
for (i = 1; i < nx; i++)
{
idata[i] = idata[i - 1] + ny;
}
return idata;
}
double **double2Dmalloc(int nx, int ny)
{
int i;
double **ddata;
ddata = (double **)malloc(nx * sizeof(double *) + nx * ny * sizeof(double));
ddata[0] = (double *)(ddata + nx);
for (i = 1; i < nx; i++)
{
ddata[i] = ddata[i - 1] + ny;
}
return ddata;
}
