
#include<omp.h>
#include<stdio.h>
#include<time.h>
#include<ctime>
#include <cstdlib>
#include <stdlib.h>


void QuickSortInter(int A[], int l, int r);
void QuickSortPara(int A[], int l, int r, int numThreads);
void Quicksort(int A[], int l, int r);


void QuickSortPara(int A[], int l, int r, int numThreads)
{
#pragma omp parallel num_threads(numThreads) 
{

#pragma omp single nowait
{
QuickSortInter(A, l, r);
}


}
}

void QuickSortInter(int A[], int l, int r)
{

if (l >= r) return;
int i = l;
int j = r;
int x = A[(l + r) / 2];
while (i <= j)
{
while (A[i] < x)
{
i++;
}
while (A[j] > x)
{
j--;
}
if (i <= j)
{
int temp = A[i];
A[i] = A[j];
A[j] = temp;
i++;
j--;
}
}
if ((r - l)<1000)
{
QuickSortInter(A, l, j);
QuickSortInter(A, i, r);
}
else
{
#pragma omp task
{
QuickSortInter(A, l, j);
}

#pragma omp task
{
QuickSortInter(A, i, r);
}
}
}

void Quicksort(int A[], int l, int r)
{
if (l >= r) return;
int i = l;
int j = r;
int x = A[(l + r) / 2];
while (i <= j)
{
while (A[i] < x)
{
i++;
}
while (A[j] > x)
{
j--;
}
if (i <= j)
{
int temp = A[i];
A[i] = A[j];
A[j] = temp;
i++;
j--;
}
}
Quicksort(A, l, j);
Quicksort(A, i, r);

}

int main()
{
int numThreads;
int num;
printf("Nhap so threads :");
scanf("%d", &numThreads);
printf("Nhap so phan tu :");
scanf("%d", &num);
int len = num;
printf("\nTong so threads : %d \n", numThreads);
int* A = (int*)malloc(len * sizeof(int));
int* B = (int*)malloc(len * sizeof(int));
printf("\tBAT DAU \t\n\n");
FILE *f;
f = fopen("result.text", "wt");
for (int threads = 2; threads <= numThreads; threads = threads + 2)
{
printf("Bat dau quicksort mang %d\n  Su dung %d Threads \n", len, threads);
fprintf(f, "Bat dau quicksort mang %d\n  Su dung %d Thread \n", len, threads);

for (int i = 0; i < len; i++)
{
int cc = rand();
A[i] = B[i] = rand() + i*cc + rand() / 2 + 1 + rand() / 3 + cc*i / 2;
}
printf("\tNap xong mang\n");
double begina = omp_get_wtime();
QuickSortPara(A, 0, len - 1, threads);
double beginb = omp_get_wtime();
printf("\nThoi gian chay song song %f time \n", (beginb - begina));

fprintf(f, "\nThoi gian chay song song %f time \n", (beginb - begina));

double begin = clock();
Quicksort(B, 0, len - 1);
double end = clock();
printf("\nThoi gian QuickSort %f seconds \n\n", (end - begin) / CLOCKS_PER_SEC);
fprintf(f, "\nThoi gian QuickSort %f seconds \n\n", (end - begin) / CLOCKS_PER_SEC);

printf("\t===================================================\t\n\n");
fprintf(f, "\t===================================================\t\n\n");
}
fclose(f);
printf("\tXONG\t");
free(B);
free(A);
return 1;
}