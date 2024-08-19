#include <stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <cstdlib>


int main() {
int NoofRows = 4, NoofCols = 4;
int i, j;
int **Matrix, **Trans, **Checkoutput;
float flops;

struct timeval  TimeValue_Start;
struct timezone TimeZone_Start;

struct timeval  TimeValue_Final;
struct timezone TimeZone_Final;
long time_start, time_end, time_overhead;




Matrix = (int **) calloc(100, sizeof(int) * NoofRows);
for (i = 0; i < NoofRows; i++) {
Matrix[i] = (int *) calloc(100, sizeof(int) * NoofCols);
for (j = 0; j < NoofCols; j++) {
printf("i: %d, j: %d ", i, j);
Matrix[i][j] = (i * j) * 1 + i;
printf("value: %d \n", Matrix[i][j]);
}
}



Trans = (int **) calloc(100, sizeof(int) * NoofCols);
Checkoutput = (int **) calloc(100, sizeof(int) * NoofCols);



for (i = 0; i < NoofCols; i++) {
Checkoutput[i] = (int *) calloc(100, sizeof(int) * NoofRows);
Trans[i] = (int *) calloc(100, sizeof(int) * NoofRows);
for (j = 0; j < NoofRows; j++) {
Checkoutput[i][j] = 0;
Trans[i][j] = 0;
}
}

gettimeofday(&TimeValue_Start, &TimeZone_Start);



#pragma omp parallel for private(j)
for (i = 0; i < NoofRows; i = i + 1)
for (j = 0; j < NoofCols; j = j + 1)
Trans[j][i] = Matrix[i][j];

gettimeofday(&TimeValue_Final, &TimeZone_Final);

time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
time_overhead = time_end - time_start;




for (i = 0; i < NoofRows; i = i + 1)
for (j = 0; j < NoofCols; j = j + 1)
Checkoutput[j][i] = Matrix[i][j];

for (i = 0; i < NoofCols; i = i + 1)
for (j = 0; j < NoofRows; j = j + 1)
if (Checkoutput[i][j] == Trans[i][j])
continue;
else {
printf("There Is A Difference From Serial And Parallel Calculation \n");
}

printf("\nTime Overhead = %ld\n", time_overhead);

printf("The Input Matrix Is \n");

for (i = 0; i < NoofRows; i++) {
for (j = 0; j < NoofCols; j++)
printf("%d \t", Matrix[i][j]);
printf("\n");
}

printf("\nThe Transpose Matrix Is \n");
for (i = 0; i < NoofCols; i = i + 1) {
for (j = 0; j < NoofRows; j = j + 1)
printf("%d \t", Trans[i][j]);
printf("\n");
}



flops = (float) 2 * NoofRows * NoofCols / (float) time_overhead;



free(Matrix);
free(Checkoutput);
free(Trans);
}