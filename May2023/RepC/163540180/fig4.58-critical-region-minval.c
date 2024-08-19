#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#define TRUE  1
#define FALSE 0
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
#define N 10
int main()
{
int    ix;
double Scale, LScale, ssq, Lssq;
#ifdef _OPENMP
(void) omp_set_dynamic(FALSE);
if (omp_get_dynamic()) {printf("Warning: dynamic adjustment of threads has been set\n");}
(void) omp_set_num_threads(3);
#endif
#pragma omp parallel 
{
#pragma omp single
printf("Number of threads is %d\n",omp_get_num_threads());
}
Scale = 2.0;
ssq   = 1.0;
printf("Before parallel region: Scale = %f ssq = %f\n",Scale,ssq);
#pragma omp parallel default(none) private(ix,LScale,Lssq) shared(Scale,ssq)
{
int TID = omp_get_thread_num();
Lssq = 2.0 * TID;
#pragma omp for
for (ix = 1; ix < N; ix ++)
{
LScale = TID + ix;
}
printf("Thread %d computed LScale = %f\n",TID,LScale);
#pragma omp critical
{
printf("Thread %d entered critical region\n",TID);
if ( Scale < LScale )
{
ssq = (Scale/LScale) * ssq + Lssq;
Scale = LScale;
printf("\tThread %d: Reset Scale to %f\n",TID,Scale);
} else {
ssq = ssq + (LScale/Scale) * Lssq;
} 
printf("\tThread %d: New value of ssq = %f\n",TID,ssq);
} 
} 
printf("After parallel region: Scale = %f ssq = %f\n",Scale,ssq);
return(0);
}
