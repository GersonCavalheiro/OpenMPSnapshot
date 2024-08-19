#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
int num_threads=1;
void write_output(char fname[], double** arr, int n )
{
FILE *f = fopen(fname, "w");
for( int i = 0; i < n; i++){
for(int j = 0; j < n; j++){
fprintf(f, "%0.12f ", arr[i][j]);
}
fprintf(f, "\n");
}
fclose(f);
}
void crout_0(double **A, double **L, double **U, int n) {
int i, j, k;
double sum = 0;
for (i = 0; i < n; i++) {
U[i][i] = 1;
}
for (j = 0; j < n; j++) {
for (i = j; i < n; i++) {
sum = 0;
for (k = 0; k < j; k++) {
sum = sum + L[i][k] * U[k][j];
}
L[i][j] = A[i][j] - sum;
}
for (i = j; i < n; i++) {
sum = 0;
for(k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][i];
}
if (L[j][j] == 0) {
exit(0);
}
U[j][i] = (A[j][i] - sum) / L[j][j];
}
}
}
void crout_1(double **A, double **L, double **U, int n) {
int i, j, k;
double sum = 0;
for (i = 0; i < n; i++) {
U[i][i] = 1;
}
for (j = 0; j < n; j++) {
#pragma omp parallel for private(i,k,sum)
for (i = j; i < n; i++) {
sum = 0;
for (k = 0; k < j; k++) {
sum = sum + L[i][k] * U[k][j];
}
L[i][j] = A[i][j] - sum;
}
#pragma omp parallel for private(i,k,sum)
for (i = j; i < n; i++) {
sum = 0;
for(k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][i];
}
if (L[j][j] == 0) {
exit(0);
}
U[j][i] = (A[j][i] - sum) / L[j][j];
}
}
}
void crout_2(double **A, double **L, double **U, int n) {
int i, j, k;
double sum = 0;
for (i = 0; i < n; i++) {
U[i][i] = 1;
}
for (j = 0; j < n; j++) {
sum = 0;
for (k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][j];
}
L[j][j] = A[j][j] - sum;
int x=j+(n-j)/2;
#pragma omp parallel sections
{
#pragma omp section
{
for (int i = j+1; i < x; i++) {
double sum = 0;
for (int k = 0; k < j; k++) {
sum = sum + L[i][k] * U[k][j];
}
L[i][j] = A[i][j] - sum;
}
}
#pragma omp section
{
for (int i =x; i < n; i++) {
double sum = 0;
for (int k = 0; k < j; k++) {
sum = sum + L[i][k] * U[k][j];
}
L[i][j] = A[i][j] - sum;
}
}
#pragma omp section
{
for (int i = j; i < x; i++) {
double sum = 0;
for(int k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][i];
}
if (L[j][j] == 0) {
exit(0);
}
U[j][i] = (A[j][i] - sum) / L[j][j];
}
}
#pragma omp section
{
for (int i = x; i < n; i++) {
double sum = 0;
for(int k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][i];
}
if (L[j][j] == 0) {
exit(0);
}
U[j][i] = (A[j][i] - sum) / L[j][j];
}
}
}
}
}
void crout_3(double **A, double **L, double **U, int n) {
int i, j, k;
double sum = 0;
for (i = 0; i < n; i++) {
U[i][i] = 1;
}
for (j = 0; j < n; j++) {
sum = 0;
for (k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][j];
}
L[j][j] = A[j][j] - sum;
#pragma omp parallel sections
{
#pragma omp section
{
#pragma omp parallel for schedule(static) private(i,k,sum)
for (i = j+1; i < n; i++) {
sum = 0;
for (k = 0; k < j; k++) {
sum = sum + L[i][k] * U[k][j];
}
L[i][j] = A[i][j] - sum;
}
}
#pragma omp section
{
#pragma omp parallel for schedule(static) private(i,k,sum)
for (i = j; i < n; i++) {
sum = 0;
for(k = 0; k < j; k++) {
sum = sum + L[j][k] * U[k][i];
}
if (L[j][j] == 0) {
exit(0);
}
U[j][i] = (A[j][i] - sum) / L[j][j];
}
}
}
}
}
int main(int argc, char const *argv[]) {
int n=atoi(argv[1]);
const char* filename=argv[2];
num_threads=atoi(argv[3]);
omp_set_num_threads(num_threads);
int strategy=atoi(argv[4]);
double **A=(double **)malloc(sizeof(double*)*n);
for (size_t i = 0; i < n; i++) {
A[i]=(double*)malloc(sizeof(double)*n);
}
double **L=(double **)malloc(sizeof(double*)*n);
for (size_t i = 0; i < n; i++) {
L[i]=(double*)malloc(sizeof(double)*n);
}
double **U=(double **)malloc(sizeof(double*)*n);
for (size_t i = 0; i < n; i++) {
U[i]=(double*)malloc(sizeof(double)*n);
}
FILE* fileA=fopen(filename,"r");
if (fileA==NULL) {
printf("Error opening the input file for matrix A. exiting\n" );
exit(-1);
}
for (size_t i = 0; i < n; i++) {
for (size_t j = 0; j < n; j++) {
fscanf(fileA,"%lf",&A[i][j]);
L[i][j]=0;
U[i][j]=0;
}
}
switch (strategy) {
case 0:
crout_0(A,L,U,n);
break;
case 1:
crout_1(A,L,U,n);
break;
case 2:
crout_2(A,L,U,n);
break;
case 3:
crout_3(A,L,U,n);
break;
}
char * outL=malloc(sizeof(char)*17+1);
char * outU=malloc(sizeof(char)*17+1);
strcpy(outL,"output_L_");
strcpy(outU,"output_U_");
outL[9]='\0';
outU[9]='\0';
strcat(outL,argv[4]);
strcat(outU,argv[4]);
strcat(outL,"_");
strcat(outU,"_");
strcat(outL,argv[3]);
strcat(outU,argv[3]);
strcat(outL,".txt");
strcat(outU,".txt");
write_output(outL, L, n );
write_output(outU, U, n );
free(outL);
free(outU);
for (size_t i = 0; i < n; i++) {
free(A[i]);
free(L[i]);
free(U[i]);
}
free(A);
free(L);
free(U);
return 0;
}
