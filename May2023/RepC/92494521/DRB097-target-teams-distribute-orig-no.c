#include <stdio.h>
#define min(x, y) (((x) < (y)) ? (x) : (y))
int main(int argc, char* argv[])
{
int i, i2;
int len = 2560;
double sum =0.0, sum2=0.0;
double a[len], b[len];
for (i=0; i<len; i++)
{
a[i]= ((double)i)/2.0;
b[i]= ((double)i)/3.0;
}
#pragma omp target map(to: a[0:len], b[0:len]) map(tofrom: sum)
#pragma omp teams num_teams(10) thread_limit(256) reduction (+:sum) 
#pragma omp distribute
for (i2=0; i2< len; i2+=256)  
#pragma omp parallel for reduction (+:sum)
for (i=i2;i< min(i2+256, len); i++)
sum += a[i]*b[i];
#pragma omp parallel for reduction (+:sum2)
for (i=0;i< len; i++)
sum2 += a[i]*b[i];
printf ("sum=%f sum2=%f\n", sum, sum2);
return 0;
}
