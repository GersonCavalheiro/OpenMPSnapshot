#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<omp.h>
void count_serial(int a[], int n)
{ 
int i, j, count;
int* temp = malloc(n*sizeof(int));
for (i = 0; i < n; i++)
{ 
count = 0; 
for (j = 0; j < n; j++) 
if (a[j] < a[i]) count++; 
else if (a[j] == a[i] && j < i)
count++;
temp[count] = a[i]; 
} 
memcpy(a, temp, n*sizeof(int));
free(temp); 
}
void count(int a[], int n)
{
int i, j, count;
int* temp = malloc(n*sizeof(int));
#pragma omp parallel for private(j, count)
for (i = 0; i < n; i++)
{ 
count = 0; 
for (j = 0; j < n; j++) 
if (a[j] < a[i]) count++; 
else if (a[j] == a[i] && j < i)
count++;
temp[count] = a[i]; 
} 
#pragma omp parallel for 
for (i = 0; i < n; i++)
{
a[i] = temp [i];
}
free(temp); 
return;
}
int cmpfunc(const void * a, const void * b)
{
return (*(int *) a - *(int *) b);
}
int main()
{
int n = 1000, temp;
int *a, *b, *c;
double ta, tb, tc;
clock_t start, end;
srand(time(NULL));
a = malloc(n*sizeof(int));
b = malloc(n*sizeof(int));
c = malloc(n*sizeof(int));
for(int i = 0; i< n; i++)
{
temp = rand()%10000;
a[i] = b[i] = c[i] = temp;
}
start = clock();
qsort(a, n, sizeof(int), cmpfunc);
end = clock();
ta = end - start;
printf("The time of qsort to sort %d numbers is %fs\n", n, ta/CLOCKS_PER_SEC);
start = clock();
count_serial(b, n);
end = clock();
tb = end - start;
printf("The time of serial count sort for sort %d numbers is %fs\n", n, tb/CLOCKS_PER_SEC);
start = clock();
count(c, n);
end = clock();
tc = end - start;
printf("The time of parallel count sort for sort %d numbers is %fs\n", n, tc/CLOCKS_PER_SEC);
return 0;
}