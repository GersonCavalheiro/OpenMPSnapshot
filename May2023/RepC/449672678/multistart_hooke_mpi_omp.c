
































































































































#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h> 
#include <mpi.h>

#define MAXVARS (250)	
#define RHO_BEGIN (0.5) 
#define EPSMIN (1E-6)	
#define IMAX (5000)		


unsigned long funevals = 0;


double f(double *x, int n)
{
double fv;
int i;

funevals++;
fv = 0.0;
for (i = 0; i < n - 1; i++) 
fv = fv + 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((x[i] - 1.0), 2);

return fv;
}


double best_nearby(double delta[MAXVARS], double point[MAXVARS], double prevbest, int nvars)
{
double z[MAXVARS];
double minf, ftmp;
int i;
minf = prevbest;
for (i = 0; i < nvars; i++)
z[i] = point[i];
for (i = 0; i < nvars; i++)
{
z[i] = point[i] + delta[i];
ftmp = f(z, nvars);
if (ftmp < minf)
minf = ftmp;
else
{
delta[i] = 0.0 - delta[i];
z[i] = point[i] + delta[i];
ftmp = f(z, nvars);
if (ftmp < minf)
minf = ftmp;
else
z[i] = point[i];
}
}
for (i = 0; i < nvars; i++)
point[i] = z[i];

return (minf);
}

int hooke(int nvars, double startpt[MAXVARS], double endpt[MAXVARS], double rho, double epsilon, int itermax)
{
double delta[MAXVARS];
double newf, fbefore, steplength, tmp;
double xbefore[MAXVARS], newx[MAXVARS];
int i, j, keep;
int iters, iadj;

for (i = 0; i < nvars; i++)
{
newx[i] = xbefore[i] = startpt[i];
delta[i] = fabs(startpt[i] * rho);
if (delta[i] == 0.0)
delta[i] = rho;
}
iadj = 0;
steplength = rho;
iters = 0;
fbefore = f(newx, nvars);
newf = fbefore;
while ((iters < itermax) && (steplength > epsilon))
{
iters++;
iadj++;
#if DEBUG
printf("\nAfter %5d funevals, f(x) =  %.4le at\n", funevals, fbefore);
for (j = 0; j < nvars; j++)
printf("   x[%2d] = %.4le\n", j, xbefore[j]);
#endif

for (i = 0; i < nvars; i++)
{
newx[i] = xbefore[i];
}
newf = best_nearby(delta, newx, fbefore, nvars);

keep = 1;
while ((newf < fbefore) && (keep == 1))
{
iadj = 0;
for (i = 0; i < nvars; i++)
{

if (newx[i] <= xbefore[i])
delta[i] = 0.0 - fabs(delta[i]);
else
delta[i] = fabs(delta[i]);

tmp = xbefore[i];
xbefore[i] = newx[i];
newx[i] = newx[i] + newx[i] - tmp;
}
fbefore = newf;
newf = best_nearby(delta, newx, fbefore, nvars);

if (newf >= fbefore)
break;





keep = 0;
for (i = 0; i < nvars; i++)
{
keep = 1;
if (fabs(newx[i] - xbefore[i]) > (0.5 * fabs(delta[i])))
break;
else
keep = 0;
}
}
if ((steplength >= epsilon) && (newf >= fbefore))
{
steplength = steplength * rho;
for (i = 0; i < nvars; i++)
{
delta[i] *= rho;
}
}
}
for (i = 0; i < nvars; i++)
endpt[i] = xbefore[i];

return (iters);
}

double get_wtime(void)
{
struct timeval t;

gettimeofday(&t, NULL);

return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

int main(int argc, char *argv[])
{
double startpt[MAXVARS], endpt[MAXVARS];
int itermax = IMAX;
double rho = RHO_BEGIN;
double epsilon = EPSMIN;
int nvars;
int trial, ntrials;
double fx;
int i, jj, j;
double t0, t1;

double best_fx = 1e10;
double best_pt[MAXVARS];
int best_trial = -1;
int best_jj = -1;

int myid, p; 
int source; 
int tag = 1234; 
int dest = 0; 
MPI_Status status; 

for (i = 0; i < MAXVARS; i++)
best_pt[i] = 0.0;

ntrials = 128 * 1024; 
nvars = 16;			  
t0 = get_wtime();
int provided;
MPI_Init(&argc, &argv); 
MPI_Comm_rank(MPI_COMM_WORLD, &myid); 
MPI_Comm_size(MPI_COMM_WORLD, &p);	  
srand48(time(0) * (myid + 1)); 
#pragma omp parallel num_threads(12) firstprivate(fx, jj, i) 
{
#pragma omp for 
for (trial = 0; trial < ntrials / p; trial++) 
{

for (i = 0; i < nvars; i++)
{
startpt[i] = 4.0 * drand48() - 4.0;
}

jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);
#if DEBUG
printf("\n\n\nHOOKE %d USED %d ITERATIONS, AND RETURNED\n", trial, jj);
for (i = 0; i < nvars; i++)
printf("x[%3d] = %15.7le \n", i, endpt[i]);
#endif

fx = f(endpt, nvars);
#if DEBUG
printf("f(x) = %15.7le\n", fx);
#endif
#pragma omp critical 
{
if (fx < best_fx)
{
best_trial = trial * (myid + 1);
best_jj = jj;
best_fx = fx;
for (i = 0; i < nvars; i++)
best_pt[i] = endpt[i];
}
}
}
}

if (myid != 0) 
{
MPI_Send(&best_fx, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD); 
MPI_Send(&best_jj, 1, MPI_INT, dest, tag + 1, MPI_COMM_WORLD);
MPI_Send(best_pt, MAXVARS, MPI_DOUBLE, dest, tag + 2, MPI_COMM_WORLD);
MPI_Send(&best_trial, 1, MPI_INT, dest, tag + 3, MPI_COMM_WORLD);
}
else
{
for (j = 1; j < p; j++) 
{
source = j;
MPI_Recv(&fx, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status); 
MPI_Recv(&jj, 1, MPI_INT, source, tag + 1, MPI_COMM_WORLD, &status);
MPI_Recv(endpt, MAXVARS, MPI_DOUBLE, source, tag + 2, MPI_COMM_WORLD, &status);
MPI_Recv(&trial, 1, MPI_INT, source, tag + 3, MPI_COMM_WORLD, &status);
if (fx < best_fx) 
{
best_trial = trial;
best_jj = jj;
best_fx = fx;
for (i = 0; i < nvars; i++)
best_pt[i] = endpt[i];
}
}

t1 = get_wtime();

printf("\n\nFINAL RESULTS:\n");
printf("Elapsed time = %.3lf s\n", t1 - t0);
printf("Total number of trials = %d\n", ntrials);
printf("Total number of function evaluations = %ld\n", funevals * p); 
printf("Best result at trial %d used %d iterations, and returned\n", best_trial, best_jj);
for (i = 0; i < nvars; i++)
{
printf("x[%3d] = %15.7le \n", i, best_pt[i]);
}
printf("f(x) = %15.7le\n", best_fx);
}

MPI_Finalize();

return 0;
}
