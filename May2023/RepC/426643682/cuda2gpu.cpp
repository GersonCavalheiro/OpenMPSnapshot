#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <omp.h>
#include <math.h>
#define VECT_SIZE 15000000
extern "C" double one_jacobi_iteration(int , int *,double , double ,
int , int ,
double *, double *,
double , double ,
double , double ,double ,double ,double ,double * );
double checkSolution(double xStart, double yStart,
int maxXCount, int maxYCount,
double *u,
double deltaX, double deltaY,
double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
int x, y;
double fX, fY;
double localError, error = 0.0;
for (y = 1; y < (maxYCount-1); y++)
{
fY = yStart + (y-1)*deltaY;
for (x = 1; x < (maxXCount-1); x++)
{
fX = xStart + (x-1)*deltaX;
localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
error += localError*localError;
}
}
return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}
void readInputFile(int* n, int* m, double* alpha, double* relax, double* tol, int* mits){
scanf("%d,%d", n, m);
scanf("%lf", alpha);
scanf("%lf", relax);
scanf("%lf", tol);
scanf("%d", mits);
printf( "READ n: %d, m: %d, alpha: %f, relax: %f, tol: %f, mits: %d \n", *n,*m,*alpha,*relax,*tol,*mits);
}
int main(int argc, char* argv[]) {
size_t freeCUDAMem, totalCUDAMem;
cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
printf("Total GPU memory %zu, free %zu\n", totalCUDAMem, freeCUDAMem);
int n, m, mits;
double alpha, tol, relax;
double maxAcceptableError;
double *u, *u_old, *tmp;
int allocCount;
int iterationCount;
readInputFile(&n, &m, &alpha, &relax, &tol, &mits);
allocCount = (n+2)*(m+2);
maxAcceptableError = tol;
double xLeft = -1.0, xRight = 1.0;
double yBottom = -1.0, yUp = 1.0;
double deltaX = (xRight-xLeft)/(n-1);
double deltaY = (yUp-yBottom)/(m-1);
double cx = 1.0/(deltaX*deltaX);
double cy = 1.0/(deltaY*deltaY);
double cc = -2.0*cx-2.0*cy-alpha;
iterationCount = 0;
double error = HUGE_VAL;
u = 	(double*)calloc(allocCount, sizeof(double)); 
u_old = (double*)calloc(allocCount, sizeof(double));
double totalTime = 0.0;
while (iterationCount < mits && error > maxAcceptableError)
{
error = 0.0;
omp_set_num_threads(2);
#pragma omp parallel shared(u_old,u),reduction(+:error)
{
int TotalGpuNum;
int ID =omp_get_thread_num();
int numthreads = omp_get_max_threads();
int LSIZE = (n+2) /numthreads;
error += one_jacobi_iteration(ID, &TotalGpuNum,xLeft, yBottom,
(n+2),(LSIZE+1),
u_old+((LSIZE-1)*(m+2)*ID), u+((LSIZE-1)*(m+2)*ID),
deltaX, deltaY,
alpha, relax,cx,cy,cc,&totalTime);
}
error = sqrt(error)/((n-2)*(m-2));
iterationCount++;
tmp = u_old;
u_old = u;
u = tmp;
}
printf( "Iterations=%3d Elapsed CUDA Wall time is %f msec\n", iterationCount, totalTime );
printf("Time taken %d seconds %5d milliseconds\n", (int)totalTime/1000, (int)totalTime%1000);
printf("Residual %g\n",error);
double absoluteError = checkSolution(xLeft, yBottom,
n+2, m+2,
u_old,
deltaX, deltaY,
alpha);
printf("The error of the iterative solution is %g\n", absoluteError);
free(u);
free(u_old);
return 0;
}
