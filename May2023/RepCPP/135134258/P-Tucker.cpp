



#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <Eigen/QR>
#include <Eigen/Dense>
#include <omp.h>
using namespace std;
using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic> MatrixXdd; 

#define lambda 0.001						
char* InputPath;
char* ResultPath;

int order;										
int *dimensionality, max_dim;					
int Entries_N;									
double *Entries;								
int *Indices;										
int *WhereX, *CountX;					        
double NormX;									
int threadsN;									


int Core_N = 1;									
int *Core_size, Core_dim;						
double *CoreTensor;								
int *CorePermu;									
double MaxCore;									
int *WhereG, *CountG;							

double *FactorM; 


double Fit, pFit = -1;
double *Error_T;
double Error, RMSE, MAE, Test_RMSE;
int *crows, rowcount;
double *tempCore;
int *Mul, *tempPermu;
int i, j, k, l;


double frand(double x, double y) {
return ((y - x)*((double)rand() / RAND_MAX)) + x;
}

double abss(double x) {
return x > 0 ? x : -x;
}

void assign_index() {
int *tempX = (int *)malloc(sizeof(int)*max_dim*order);
int pos = 0, i, j, k, l;
for (i = 0; i < order; i++) {
for (j = 0; j < dimensionality[i]; j++) {
CountX[i*max_dim + j] = tempX[i*max_dim + j] = 0;
}
}
for (i = 0; i < Entries_N; i++) {
for (j = 0; j < order; j++) {
k = Indices[pos++];
CountX[j*max_dim + k]++;
tempX[j*max_dim + k]++;
}
}
pos = 0;
int now = 0;
for (i = 0; i < order; i++) {
pos = i*max_dim;
for (j = 0; j < dimensionality[i]; j++) {
k = CountX[pos];
CountX[pos] = now;
tempX[pos++] = now;
now += k;
}
CountX[pos] = now;
tempX[pos] = now;
}
pos = 0;
for (i = 0; i < Entries_N; i++) {
for (j = 0; j < order; j++) {
k = Indices[pos++];
int now = tempX[j*max_dim + k];
WhereX[now] = i;
tempX[j*max_dim + k]++;
}
}
}
char tmp[10005];
void Getting_Input() {
FILE *fin = fopen(InputPath, "r");
FILE *fin2 = fopen(InputPath, "r");
FILE *ftest = fopen(ResultPath, "r");

double Timee = clock();

printf("Reading Input...\n");

dimensionality = (int *)malloc(sizeof(int)*order);
Core_size = (int *)malloc(sizeof(int)*order);

for (i = 0; i < order; i++) {
dimensionality[i] = 0;
Core_size[i] = Core_dim;
Core_N *= Core_size[i];
}

int len = 0;
while (fgets(tmp, 10005, fin)) {
Entries_N++;
}
Indices = (int *)malloc(sizeof(int)*Entries_N*order);
Entries = (double *)malloc(sizeof(double)*Entries_N);
int pos = 0;
for (i = 0; i < Entries_N; i++) {
fgets(tmp, 10005, fin2);
len = strlen(tmp);
int k = 0, idx = 0, flag = 0, flag2 = 0;
double mul = 0.1, val = 0;
for (j = 0; j < len; j++) {
if (tmp[j] == ' ' || tmp[j] == '\t') {
Indices[pos++] = idx;
if (dimensionality[k] < idx+1) dimensionality[k] = idx+1;
idx = 0;
k++;
}
else if (tmp[j] >= '0' && tmp[j] <= '9') {
if (flag == 1) {
val += mul*(tmp[j] - '0');
mul /= 10;
}
else idx = idx * 10 + tmp[j] - '0';
}
else if (tmp[j] == '.') {
val += idx;
flag = 1;
}
else if (tmp[j] == '-') {
flag2 = 1;
}
}
if(flag==0) val = idx;
if(flag2 == 0){ 
Entries[i] = val;
}else{
Entries[i] = -1 * val;
}
NormX += Entries[i] * Entries[i];
}
for (i = 0; i < order; i++) {
if (max_dim < dimensionality[i]) max_dim = dimensionality[i];
}
max_dim++;
WhereX = (int *)malloc(sizeof(int)*order*Entries_N);
CountX = (int *)malloc(sizeof(int)*max_dim*order);

NormX = sqrt(NormX);
printf("Reading Done.\n\n[METADATA]\nTensor Order: %d\tSize: ", order);
for (i = 0; i < order; i++) {
if (i != order - 1) printf("%dx", dimensionality[i]);
else printf("%d\t", dimensionality[i]);
}
printf("Rank: %d\tNNZ : %d\tThreads : %d\tNorm : %lf\n\nInitialize...\n", Core_dim, Entries_N, threadsN, NormX);

assign_index();
FactorM = (double *)malloc(sizeof(double)*order*max_dim*Core_dim);
for (i = 0; i < order; i++) {
int row = dimensionality[i], col = Core_size[i];
for (j = 0; j < row; j++) {
for (k = 0; k < col; k++) {
FactorM[i*max_dim*col + j*col + k] = frand(0, 1);
}
}
}

CoreTensor = (double *)malloc(sizeof(double)*Core_N);
CorePermu = (int *)malloc(sizeof(int)*Core_N*order);
pos = 0;
for (i = 0; i < Core_N; i++) {
CoreTensor[i] = frand(0, 1);
if (i == 0) {
for (j = 0; j < order; j++) CorePermu[j] = 0;
}
else {
for (j = 0; j < order; j++) {
CorePermu[i*order + j] = CorePermu[(i - 1)*order + j];
}
CorePermu[i*order + order - 1]++;  k = order - 1;
while (CorePermu[i*order + k] >= Core_size[k]) {
CorePermu[i*order + k] -= Core_size[k];
CorePermu[i*order + k - 1]++; k--;
}
}
}
printf("Elapsed Time for I/O and Initializations:\t%lf\n\n", (clock() - Timee) / CLOCKS_PER_SEC);
}


void Update_Factor_Matrices() {

int mult = max_dim*Core_dim;
for (i = 0; i < order; i++) { 
int row_size = dimensionality[i];
int column_size = Core_size[i];
int iter;
rowcount = 0;
crows = (int *)malloc(sizeof(int)*row_size);
for (j = 0; j < row_size; j++) { 
int pos = i*max_dim + j, nnz = CountX[pos + 1] - CountX[pos];
if (nnz != 0) {
crows[rowcount++] = j;
}
}

#pragma omp parallel for schedule(dynamic) 
for (iter = 0; iter < rowcount; iter++) {
int j, k, l, ii, jj;
j = crows[iter]; 
double *Delta = (double *)malloc(sizeof(double)*column_size);
double *B = (double *)malloc(sizeof(double)*column_size*column_size);
double *C = (double *)malloc(sizeof(double)*column_size);

int pos = 0;
for (k = 0; k < column_size; k++) {
for (l = 0; l < column_size; l++) {
B[pos] = 0;
if (k == l) B[pos] = lambda;
pos++;
}
C[k] = 0;
}

pos = i*max_dim + j;
int nnz = CountX[pos + 1] - CountX[pos];
pos = CountX[pos];
for (k = 0; k < nnz; k++) { 
int current_input_entry = WhereX[pos + k];
int pre_val = current_input_entry*order;
int *cach1 = (int *)malloc(sizeof(int)*order);
for (l = 0; l < order; l++) cach1[l] = Indices[pre_val++];
for (l = 0; l < column_size; l++) Delta[l] = 0;
for (l = 0; l < Core_N; l++) {
int pre1 = l*order, pre2 = 0;
int CorePos = CorePermu[pre1 + i];
double res = CoreTensor[l];
for (ii = 0; ii < order; ii++) {
if (ii != i) {
int mulrow = cach1[ii], mulcol = CorePermu[pre1];
res *= FactorM[pre2 + mulrow*column_size + mulcol];
}
pre1++;
pre2 += mult;
}
Delta[CorePos] += res;
}
free(cach1);
int now = 0;
double Entry_val = Entries[current_input_entry];
for (ii = 0; ii < column_size; ii++) {
double cach = Delta[ii];
for (jj = 0; jj < column_size; jj++) {
B[now++] += cach * Delta[jj];
}
C[ii] += cach * Entry_val;
}
}
free(Delta);
MatrixXdd A(column_size, column_size);
pos = 0;
for (k = 0; k < column_size; k++) {
for (l = 0; l < column_size; l++) {
A(k, l) = B[k*column_size + l];
}
}
MatrixXdd BB = A.inverse();
pos = 0;
for (k = 0; k < column_size; k++) {
for (l = 0; l < column_size; l++) {
B[k*column_size + l] = BB(k, l);
}
}

int cach = i*mult + j*column_size;
for (k = 0; k < column_size; k++) {
double res = 0;
for (l = 0; l < column_size; l++) {
res += C[l] * B[l*column_size + k];
}
FactorM[cach + k] = res;
}
A.resize(0,0); BB.resize(0,0);
free(B);
free(C);
}
free(crows);
}
}


void Reconstruction() {
RMSE = Error = 0;
Error_T = (double *)malloc(sizeof(double)*Entries_N);
#pragma omp parallel for schedule(static)
for (i = 0; i < Entries_N; i++) {
Error_T[i] = Entries[i];
}
int mult = max_dim*Core_dim;
#pragma omp parallel for schedule(static)
for (i = 0; i < Entries_N; i++) {
int j, pre_val = i*order;
double ans = 0;
int *cach1 = (int *)malloc(sizeof(int)*order);
for (j = 0; j < order; j++) cach1[j] = Indices[pre_val++];
for (j = 0; j < Core_N; j++) {
double temp = CoreTensor[j];
int k;
int pos = j*order;
int val = 0;
for (k = 0; k < order; k++) {
int mulrow = cach1[k], mulcol = CorePermu[pos++];
temp *= FactorM[val + mulrow*Core_dim + mulcol];
val += mult;
}
ans += temp;
}
free(cach1);
Error_T[i] -= ans;
}
#pragma omp parallel for schedule(static) reduction(+:Error)
for (i = 0; i < Entries_N; i++) {
Error += Error_T[i] * Error_T[i];
}
RMSE = sqrt(Error / Entries_N);
if (NormX == 0) Fit = 1;
else Fit = 1 - sqrt(Error) / NormX;
free(Error_T);
}

void Orthogonalize() {
Mul = (int *)malloc(sizeof(int)*order);
Mul[order - 1] = 1;
for (i = order - 2; i >= 0; i--) {
Mul[i] = Mul[i + 1] * Core_size[i + 1];
}
int pos = 0;
for (i = 0; i < order; i++) {
MatrixXdd X(dimensionality[i], Core_size[i]), thinQ(MatrixXdd::Identity(dimensionality[i], Core_size[i]));
for (k = 0; k < dimensionality[i]; k++) {
for (l = 0; l < Core_size[i]; l++) {
X(k, l) = FactorM[i*max_dim*Core_dim + k*Core_dim + l];
}
}
HouseholderQR<MatrixXdd> qr(X);
thinQ = qr.householderQ() * thinQ;

for (k = 0; k < dimensionality[i]; k++) {
for (l = 0; l < Core_size[i]; l++) {
FactorM[i*max_dim*Core_dim + k*Core_dim + l] = thinQ(k, l);
}
}

MatrixXdd R = thinQ.transpose()*X;
tempCore = (double *)malloc(sizeof(double)*Core_N);
tempPermu = (int *)malloc(sizeof(int)*order);
for (j = 0; j < Core_N; j++) {
tempCore[j] = 0;
}
for (j = 0; j < Core_N; j++) {
for (k = 0; k <= i - 1; k++) {
tempPermu[k] = CorePermu[j*order + k];
}
for (k = i + 1; k < order; k++) {
tempPermu[k] = CorePermu[j*order + k];
}
for (k = 0; k < Core_size[i]; k++) {
tempPermu[i] = k;
int cur = j + (k - CorePermu[j*order + i])*Mul[i];
tempCore[cur] += CoreTensor[j] * R(k, CorePermu[j*order + i]);
}
}
for (j = 0; j < Core_N; j++) {
CoreTensor[j] = tempCore[j];
}
}
}

void PTucker() {
printf("P-Tucker START!\n\n");

double Stime = omp_get_wtime();
int iter = 0;
double avertime = 0;

while (1) {

double itertime = omp_get_wtime(), steptime;
steptime = itertime;

printf("[Iteration %d]\n",++iter);

Update_Factor_Matrices();
printf("Elapsed Time for Updating Factor Matrices:\t%lf\n", omp_get_wtime() - steptime);
steptime = omp_get_wtime();

Reconstruction();
printf("Elapsed Time for Calculating Recon. Error:\t%lf\n", omp_get_wtime() - steptime);
steptime = omp_get_wtime();

avertime += omp_get_wtime() - itertime;
printf("Fit:\t%lf\nTraining RMSE:\t%lf\nElapsed Time:\t%lf\n\n", Fit, RMSE, omp_get_wtime() - itertime);

if (iter>=20 || (pFit != -1 && abss(pFit - Fit) <= 0.0001)) break;
pFit = Fit;
}

avertime /= iter;



printf("\nMain iterations are ended.\tFinal Fit : %lf\tAverage iteration time : %lf\tTotal Elapsed time: %lf\n\n", Fit, avertime,omp_get_wtime()-Stime);

}


void Print() {
printf("\nWriting factor matrices and the core tensor...\n");
char temp[10005];
int pos = 0;
int mult = max_dim*Core_dim;
for (i = 0; i < order; i++) {
sprintf(temp, "%s/FACTOR%d", ResultPath, i);
FILE *fin = fopen(temp, "w");
for (j = 0; j < dimensionality[i]; j++) {
for (k = 0; k < Core_size[i]; k++) {
if(k==Core_size[i]-1) fprintf(fin, "%e", FactorM[i*mult + j*Core_size[i] + k]);
else fprintf(fin, "%e\t", FactorM[i*mult + j*Core_size[i] + k]);
}
fprintf(fin, "\n");
}
}
sprintf(temp, "%s/CORETENSOR", ResultPath);
FILE *fcore = fopen(temp, "w");
pos = 0;
for (i = 0; i < Core_N; i++) {
for (j = 0; j < order; j++) {
fprintf(fcore, "%d\t", CorePermu[pos++]);
}
fprintf(fcore, "%e\n", CoreTensor[i]);
}
}


int main(int argc, char* argv[]) {

if (argc == 6) {

InputPath = argv[1];
ResultPath = argv[2];
order = atoi(argv[3]);
Core_dim = atoi(argv[4]);
threadsN = atoi(argv[5]);

srand((unsigned)time(NULL));

Getting_Input();

PTucker();

Print();

}

else printf("ERROR: Invalid Arguments\n\nUsage: ./P-Tucker [input_tensor_path] [result_directory_path] [tensor_order] [tensor_rank] [number of threads]\ne.g.) ./P-Tucker input.txt result/ 3 10 20\n");
return 0;
}
