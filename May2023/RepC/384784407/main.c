#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX_VALUE 20
#define SEED 13
#define MAX_PRINT 50
void printVector(int *data, int r);
void printMatrix(int **data, int r, int c);
int **populateMatrix(int r, int c);
int *populateDataArray(int n, int isRandom);
int getRandomInt(int maxValue);
int *matXvett(int **A, int *x, int rows, int col);
int main(int argc, char **argv){
int rows, columns;
int **matrix = NULL, *x = NULL;
int *b = NULL;
struct timeval time;
double start = 0, end = 0;
if(argc < 3){
printf("Missing input arguments. > /Main rows columns\n");
return -1;
}
rows = atoi(argv[1]);
columns = atoi(argv[2]);
if(rows < 1 || columns < 1){
printf("Rows (and columns) must be >= 1.\n");
return -1;
}
matrix = populateMatrix(rows, columns);
if(matrix == NULL){
return -1;
}
if((rows <= MAX_PRINT) && (columns <= MAX_PRINT)){
printf("A =\n");
printMatrix(matrix, rows, columns);
printf("\n");
}
x = populateDataArray(columns, 0);
if(x == NULL){
return -1;
}
if(columns <= MAX_PRINT){
printf("x =");
printVector(x, columns);
printf("\n");
}
gettimeofday(&time, NULL);
start = time.tv_sec + (time.tv_usec / 1000000.0);
b = matXvett(matrix, x, rows, columns);
gettimeofday(&time, NULL);
end = time.tv_sec + (time.tv_usec / 1000000.0);
if(b == NULL){
return -2;
}
if(columns <= MAX_PRINT){
printf("Results = ");
printVector(b, columns);
printf("\n");
}
printf("Matrix A = %d x %d, vector x = %d\n", rows, columns, columns);
printf("Program ended in %e seconds.\n", end - start);
return 0;
}
int *matXvett(int **A, int *x, int rows, int col){
int i, j;
int *b = (int *) calloc(rows, sizeof(int));
if(b != NULL){
#pragma omp parallel for default (none) shared (A, x, b, rows, col) private (i,j)
for (i = 0; i < rows; i++){
for (j = 0; j < col; j++){
b[i] += (A[i][j] * x[j]);
}
}
}
return b;
}
void printVector(int *data, int r){
int i;
for(i=0; i < r; i++){
printf(" %d", data[i]);
}
printf("\n");
}
void printMatrix(int **data, int r, int c){
int i, j;
for(i=0; i < r; i++){
for(j=0; j < c; j++){
printf("%d\t", data[i][j]);
}
printf("\n");
}
}
int **populateMatrix(int r, int c){
int **data = (int **) malloc(sizeof(int *) * r), i;
if(data == NULL){
printf("malloc error!\n");
return NULL;
}
for(i=0; i < r; i++){
srand(SEED + i);
data[i] = populateDataArray(c, 1);
}
return data;
}
int *populateDataArray(int n, int isRandom){
int *data = (int *) malloc(sizeof(int) * n), i;
if(data == NULL){
printf("malloc error!\n");
return NULL;
}
if(isRandom == 1){
for(i=0; i < n; i++){
data[i] = getRandomInt(MAX_VALUE);
}
} else {
for(i=0; i < n; i++){
data[i] = (i%9) + 1;
}
}
return data;
}
int getRandomInt(int maxValue) {
return (rand() % maxValue) + 1;   
}
