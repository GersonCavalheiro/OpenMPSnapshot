#include<stdio.h>
#include<omp.h>
void main()
{
int n = 10;
int x[n], y[n];	
int i,j;
for (i=0; i< n; i++) {
y[i] = i*2;
}
#pragma omp parallel default(none) shared(x,y,n) private(i)
{
#pragma omp for
for(i=0 ; i  < n  ; i ++) {
x[i] = -1 * y[i];
printf ("#\n");
}
}
for (i=0; i< n; i++) {
printf ("x(%d) <--> y(%d)\n", x[i],y[i]);
}
printf("All Done\n");
}
