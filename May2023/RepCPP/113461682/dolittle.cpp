#include<bits/stdc++.h>
#include<omp.h>
#define ld long double
#define n 2000
#define chunk 10
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

ld a[n][n],lwr[n][n],upr[n][n];
int main(){
struct timeval start, stop;  
struct timezone tz;
clock_t start2, stop2;  
unsigned long long usecstart, usecstop ;
struct tms cputstart, cputstop;  
int i,j,k,nthreads,tid;


srand(time(NULL));
for(i=0;i<n;i++)
{
for(j=0;j<n;j++)
{
a[i][j]=1+rand()%1000;
}
}


printf("\nStarting clock for parallel algorithm.\n");
gettimeofday(&start, &tz);
start2 = times(&cputstart);
#pragma omp parallel shared(lwr,upr,a,nthreads) private(tid,i,k,j)
{
tid=omp_get_thread_num();
if(tid==0)
nthreads=omp_get_num_threads();

for(k=0;k<n;k++)
{
lwr[k][k]=1;
#pragma omp for schedule(static,chunk)
for(j=k;j<n;j++)
{
ld sum=0;
for(int s=0;s<=k-1;s++)
{
sum+=lwr[k][s]*upr[s][j];
}
upr[k][j]=a[k][j]-sum;
}
#pragma omp for schedule(static,chunk)
for(i=k+1;i<n;i++)
{
ld sum=0;
for(int s=0;s<=k-1;s++)
{
sum+=lwr[i][s]*upr[s][k];
}
lwr[i][k]=(a[i][k]-sum)/upr[k][k];
}
}	
}

gettimeofday(&stop, &tz);
stop2 = times(&cputstop);
printf("Stopped clock.\n");
usecstart = (unsigned long long)start.tv_sec * 1000000 + start.tv_usec;
usecstop = (unsigned long long)stop.tv_sec * 1000000 + stop.tv_usec;

printf("Total elapsed time = %g ms.\n",(float)(usecstop-usecstart)/(float)CLOCKS_PER_SEC * 1000) ;
printf("Total elapsed time by system in calling and executing = %g ms.\n",(float)(stop2-start2) / (float)CLOCKS_PER_SEC * 1000) ;
printf("My total CPU time for parent = %g ms.\n", (float)( (cputstop.tms_utime + cputstop.tms_stime) - (cputstart.tms_utime + cputstart.tms_stime) ) /(float)CLOCKS_PER_SEC * 1000);
printf("My total CPU time for child processes = %g ms.\n", (float)( (cputstop.tms_cutime + cputstop.tms_cstime) - (cputstart.tms_cutime + cputstart.tms_cstime) ) / (float)CLOCKS_PER_SEC * 1000);
printf("--------------------------------------------\n");


printf("\nStarting clock for sequential algorithm.\n");
gettimeofday(&start, &tz);
start2 = times(&cputstart);
for(k=0;k<n;k++)
{
lwr[k][k]=1;
for(j=k;j<n;j++)
{
ld sum=0;
for(int s=0;s<=k-1;s++)
{
sum+=lwr[k][s]*upr[s][j];
}
upr[k][j]=a[k][j]-sum;
}
for(i=k+1;i<n;i++)
{
ld sum=0;
for(int s=0;s<=k-1;s++)
{
sum+=lwr[i][s]*upr[s][k];
}
lwr[i][k]=(a[i][k]-sum)/upr[k][k];
}
}

gettimeofday(&stop, &tz);
stop2 = times(&cputstop);
printf("Stopped clock.\n");
usecstart = (unsigned long long)start.tv_sec * 1000000 + start.tv_usec;
usecstop = (unsigned long long)stop.tv_sec * 1000000 + stop.tv_usec;

printf("Total elapsed time = %g ms.\n",(float)(usecstop-usecstart)/(float)CLOCKS_PER_SEC * 1000) ;
printf("Total elapsed time by system in calling and executing = %g ms.\n",(float)(stop2-start2) / (float)CLOCKS_PER_SEC * 1000) ;
printf("My total CPU time for parent = %g ms.\n", (float)( (cputstop.tms_utime + cputstop.tms_stime) - (cputstart.tms_utime + cputstart.tms_stime) ) /(float)CLOCKS_PER_SEC * 1000);
printf("My total CPU time for child processes = %g ms.\n", (float)( (cputstop.tms_cutime + cputstop.tms_cstime) - (cputstart.tms_cutime + cputstart.tms_cstime) ) / (float)CLOCKS_PER_SEC * 1000);
printf("--------------------------------------------\n");


}
