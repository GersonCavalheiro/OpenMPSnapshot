#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include<stdbool.h>
#define MAXITER 2000
#define NPOINTS 500
struct complex
{
double real;
double imag;
};
int main(int argc, char *argv[])
{
int currentProcessID, countP;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &currentProcessID);
MPI_Comm_size(MPI_COMM_WORLD, &countP);
int i, j, iter, numoutside, count = 0;
double area, error, ztemp;
double start, finish;
bool stop = false;
struct complex z, c;
start = MPI_Wtime();
#pragma omp parallel for private(ztemp, c, z, stop) schedule(guided)
for (i = currentProcessID; i < NPOINTS; i += countP){
for (j = 0; j < NPOINTS; j++){
c.real = -2.0 + 2.5 * (double)(i) / (double)(NPOINTS) + 1.0e-7;
c.imag = 1.125 * (double)(j) / (double)(NPOINTS) + 1.0e-7;
z = c;
for (iter = 0; iter < MAXITER; iter++){
ztemp = (z.real * z.real) - (z.imag * z.imag) + c.real;
z.imag = z.real * z.imag * 2 + c.imag;
z.real = ztemp;
if ((z.real * z.real + z.imag * z.imag) > 4.0e0){
#pragma omp atomic
count++;
stop = true;
break;
}
}
stop = false;
}
}
MPI_Reduce(&count, &numoutside, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
finish = MPI_Wtime();
if (currentProcessID == 0){
area = 2.0 * 2.5 * 1.125 * (double)(NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
error = area / (double)NPOINTS;
printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
printf("Time = %12.8f seconds\n", finish - start);
}
MPI_Finalize();
return 0;
}
