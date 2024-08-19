#include<stdio.h>
#include<omp.h>
int main()
{
int i=0;
int n;
printf("Done by Maitreyee\n\n");
scanf("%d", &n);
int j=0,count1=0,pc=0;
double start_time, end_time;
#pragma omp parallel for reduction(+:total)
for(i=1; i<=n; i++) {
start_time = omp_get_wtime();
int count1 = 0;
for(j=1; j<=i; j++) {
if((i%j) == 0) {
count1++;
}
}
if(count1 <= 2) {
total++;
printf("%d ", i);
end_time = omp_get_wtime()-start_time;
printf("\t %f \n", end_time);
}
}
printf("\nNumber of prime numbers between are %d\n",total);
}
