#include<stdio.h>
#include<omp.h>
int main(int argc, char* argv[])
{
int n;
printf("Number n: ");
scanf("%d", &n);
int prime[n+1];
double s1 = omp_get_wtime();
#pragma omp parallel for schedule(static)
for(int i=2; i<=n; i++)
{
if(((i!=2)&&(i!=3)&&(i!=5)&&(i!=7)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
if(((i%2!=0)&&(i%3!=0)&&(i%5!=0)&&(i%7!=0)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
prime[i]=i;
}
else
{
prime[i]=0;
}
}
}
s1 = omp_get_wtime() - s1;
printf("Time for static execution: %lf\n", s1);
double s2 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
for(int i=2; i<=n; i++)
{
if(((i!=2)&&(i!=3)&&(i!=5)&&(i!=7)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
if(((i%2!=0)&&(i%3!=0)&&(i%5!=0)&&(i%7!=0)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
prime[i]=i;
}
else
{
prime[i]=0;
}
}
}
s2 = omp_get_wtime() - s2;
printf("Time for dynamic execution: %lf\n", s2);
double s3 = omp_get_wtime();
#pragma omp parallel for schedule(guided)
for(int i=2; i<=n; i++)
{
if(((i!=2)&&(i!=3)&&(i!=5)&&(i!=7)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
if(((i%2!=0)&&(i%3!=0)&&(i%5!=0)&&(i%7!=0)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
prime[i]=i;
}
else
{
prime[i]=0;
}
}
}
s3 = omp_get_wtime() - s3;
printf("Time for guided execution: %lf\n", s3);
double s4 = omp_get_wtime();
#pragma omp parallel for schedule(auto)
for(int i=2; i<=n; i++)
{
if(((i!=2)&&(i!=3)&&(i!=5)&&(i!=7)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
if(((i%2!=0)&&(i%3!=0)&&(i%5!=0)&&(i%7!=0)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
prime[i]=i;
}
else
{
prime[i]=0;
}
}
}
s4 = omp_get_wtime() - s4;
printf("Time for auto execution: %lf\n", s4);
double s5 = omp_get_wtime();
#pragma omp parallel for schedule(runtime)
for(int i=2; i<=n; i++)
{
if(((i!=2)&&(i!=3)&&(i!=5)&&(i!=7)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
if(((i%2!=0)&&(i%3!=0)&&(i%5!=0)&&(i%7!=0)) || ((i==2)||(i==3)||(i==5)||(i==7)))
{
prime[i]=i;
}
else
{
prime[i]=0;
}
}
}
s5 = omp_get_wtime() - s5;
printf("Time for runtime execution: %lf\n", s5);
FILE* f = fopen("res.txt", "w");
for (int i=2; i <= n; i++)
{
if (prime[i])
{
fprintf(f, "%d\n",i);
}
}
return 0;
}
