#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
double count_sort_paralelo(double a[], int n, int n_threads) {
int i, j, count;
double *temp;
double start, end, duracao;
temp = (double *)malloc(n*sizeof(double));
start = omp_get_wtime();
#pragma omp parallel for num_threads(n_threads) default(none) shared(n, a, temp) private(i, j, count)
for (i = 0; i < n; i++) {
count = 0;
for (j = 0; j < n; j++)
if (a[j] < a[i])
count++;
else if (a[j] == a[i] && j < i)
count++;
temp[count] = a[i]; 
}
end = omp_get_wtime();
duracao = end - start;
memcpy(a, temp, n*sizeof(double));
free(temp);
return duracao;
}
int main(int argc, char * argv[]) {
int i, n, nt;
double  * a, t_s;
scanf("%d",&nt);
scanf("%d",&n);
a = (double *)malloc(n*sizeof(double));
for(i=0;i<n;i++)
scanf("%lf",&a[i]);
t_s = count_sort_paralelo(a, n, nt);
for(i=0;i<n;i++)
printf("%.2lf ",a[i]);
printf("\n");
printf("%lf\n",t_s);
return 0;
}
