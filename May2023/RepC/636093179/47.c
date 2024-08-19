#include <stdio.h>
#include <omp.h>
#include <limits.h>
int mymax(int r, int n)
{
printf("%d: r= %d, n= %d\n",omp_get_thread_num(),r,n);
if(r > n)
return(r);
return(n);
}
int main()
{
int N= 100, max= 0;
#pragma omp declare reduction (mm:int:omp_out=mymax(omp_out,omp_in)) initializer(omp_priv=INT_MIN)
#pragma omp parallel reduction(mm:max) num_threads(25)
{
#pragma omp for
for(int i= 1; i <= N; i++){
if(i > max)
max= i;
}
}
printf("Calculated max= %d\n",max);
}
