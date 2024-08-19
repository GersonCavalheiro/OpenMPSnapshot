#include <stdio.h>
#include <unistd.h>
#include "stdlib.h"
#include <math.h>
#include "mycom.h"
#include "mynet.h"
#include "mpi.h"
#include <omp.h>
int np, mp, nl, ier, lp, nt;
char pname[MPI_MAX_PROCESSOR_NAME];
MPI_Status status;
double tick, t1, t2, t3;
double a = 0;
double b = 1;
int ni = 1000000000;
double sum = 0;
double f1(double x);
double f1(double x) { return 4.0/(1.0+x*x); }
double myjob(int mp);
double myjob(int mp)
{
int n1; double a1, b1, h1, s1 = 0.0;
int num = 0;
omp_set_num_threads(nt);
n1 = ni / (np*nt);
h1 = (b - a) / (np*nt);
#pragma omp parallel 
{
a1 = a + h1 * (mp*nt+omp_get_thread_num());
if ((mp*nt+omp_get_thread_num())<np*nt-1) b1 = a1 + h1; else b1 = b;
double res = integrate(f1,a1,b1,n1);
#pragma omp critical
{
s1 += res;
num++;
}
}
while (num<nt);
return s1;
}
int MyNetInit_1(int* argc, char*** argv, int* np, int* mp,
int* nl, char* pname, double* tick)
{
int i;
int provided;
i = MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
if (i != 0){
fprintf(stderr,"MPI initialization error");
exit(i);
}
MPI_Comm_size(MPI_COMM_WORLD,np);
MPI_Comm_rank(MPI_COMM_WORLD,mp);
MPI_Get_processor_name(pname,nl);
*tick = MPI_Wtick();
sleep(1);
return 0;
}
int main(int argc, char* argv[])
{
sscanf(argv[1],"%d",&nt);
MyNetInit_1(&argc,&argv,&np,&mp,&nl,pname,&tick);  
fprintf(stderr,"Netsize: %d, process: %d, system: %s, tick=%12le\n",np,mp,pname,tick);
sleep(1);
if (np<2) {
t1 = MPI_Wtime();
sum = integrate(f1,a,b,ni);
t2 = MPI_Wtime();
t3 = t2;
} 
else {
int i; double p;
int distance = 1; 
t1 = MPI_Wtime();
sum = myjob(mp);
t2 = MPI_Wtime();
while(distance < np)
{	
if( (mp % (2*distance)) == 0) {
if (mp + distance < np) {
MPI_Recv(&p, 1, MPI_DOUBLE, mp + distance, MY_TAG, MPI_COMM_WORLD, &status);
sum = sum + p;
}
}
else if( (mp % distance) == 0)
MPI_Send(&sum, 1, MPI_DOUBLE, mp-distance, MY_TAG, MPI_COMM_WORLD);
distance *=2;
}
}
MPI_Barrier(MPI_COMM_WORLD);
t3 = MPI_Wtime();
t1 = t2 - t1;
t2 = t3 - t2;
t3 = t1 + t2;
if(mp == 0)
fprintf(stderr, "\nResult: %22.15le\nTime:%lf\n", sum, t3);
MPI_Finalize();
return 0;
}
