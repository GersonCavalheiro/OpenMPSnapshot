#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <unistd.h>
static int cacheLineSize;
int gcd(int a, int b) {
while (1) {
if (a == 0) 
return b; 
if (b == 0) 
return a; 
if (a == b) 
return a; 
if (a > b) {
a = a%b;
} else {
b = b%a;
}
}
}
int lcm(int a, int b) {
return (a*b)/gcd(a, b);
}
void printMat(float* mat, int m, int n) {
for (int i=0; i<m; i++) {
for (int j=0; j<n; j++) {
printf("%f", (*(mat+i*n+j)));
if (j==n-1) {
printf("\n");
} else {
printf(", ");
}
}
}
}
float subtractProjOnUnitVecFromRes(float* result, float* unitVec, float* generalVec, int m, int n) {
float dotProdRes = 0.0;
for (int i=0; i<m; i++) {
dotProdRes += (*(unitVec+i*n))*(*(generalVec+i*n));
}
for (int i=0; i<m; i++) {
*(result+i*n) -= dotProdRes*(*(unitVec+i*n));
}
return dotProdRes;
}
void QRfactors(float* d, float* q, float* r, int m, int n) {
for (int i=0; i<n; i++) {
for (int j=0; j<m; j++) {
*(q+j*n+i) = *(d+j*n+i);
}
for (int j=0; j<n; j++) {
if (j<i) {
float dotp = subtractProjOnUnitVecFromRes(q+i,q+j,d+i,m,n);
*(r+j*n+i) = dotp;
} else {
*(r+j*n+i) = 0.0;
}
}
float norm = 0.0;
for (int j=0; j<m; j++) {
norm += (*(q+j*n+i))*(*(q+j*n+i));
}
norm = sqrt(norm);
*(r+i*n+i) = norm;
for (int j=0; j<m; j++) {
*(q+j*n+i) /= norm;
}
}
}
void matmul(float* a, float* b, float*c, int l, int m, int n) {
int chunk = lcm(cacheLineSize, n)/n;
#pragma omp parallel for schedule(static, chunk)
for (int i=0; i<l; i++) {
for (int j=0; j<n; j++) {
float prod = 0.0;
for (int k=0; k<m; k++) {
prod += (*(a+i*m+k))*(*(b+k*n+j));
}
*(c+i*n+j) = prod;
}
}
}
void transpose(float* a, float* b, int m, int n) {
for (int i=0; i<m; i++) {
for (int j=0; j<n; j++) {
*(a+i*n+j) = *(b+j*m+i);
}
}
}
void SVD_internal(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
cacheLineSize = (int) ceil((double) sysconf(_SC_LEVEL1_DCACHE_LINESIZE))/((double)(sizeof(float)));
float DT_D_arr[N*N];
float* DT_D = DT_D_arr;
int chunk = lcm(cacheLineSize, N)/N;
#pragma omp for schedule(static, chunk)
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
float prod = 0.0;
for (int k=0; k<M; k++) {
prod += (*(D+k*N+i))*(*(D+k*N+j));
}
*(DT_D+i*N+j) = prod;
}
}
float qi_arr[N*N];
float ri_arr[N*N];
float di_odd_arr[N*N];
float ei_even_arr[N*N];
float ei_odd_arr[N*N];
float* qi = qi_arr;
float* ri = ri_arr;
float* di_even = DT_D;
float* di_odd = di_odd_arr;
float* ei_even = ei_even_arr;
float* ei_odd = ei_odd_arr;
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
if (i==j) {
*(ei_even+i*N+j) = 1.0;
} else {
*(ei_even+i*N+j) = 0.0;
}
}
}
int numIter = 0;
float error =0.0;
float* di;
float* diPlus1;
float* ei;
float* eiPlus1;
float* errorI;
float* errorIplus1;
while (numIter<10000) {
if (numIter%2==0) {
di = di_even;
diPlus1 = di_odd;
ei = ei_even;
eiPlus1 = ei_odd;
errorIplus1 = &error;
} else {
di = di_odd;
diPlus1 = di_even;
ei = ei_odd;
eiPlus1 = ei_even;
errorIplus1 = &error;
}
QRfactors(di, qi, ri, N, N);
matmul(ri, qi, diPlus1, N, N, N);
matmul(ei, qi, eiPlus1, N, N, N);
*errorIplus1 = 0.0;
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
float diff = *(diPlus1+i*N+j) - *(di+i*N+j);
*errorIplus1 += diff*diff;
diff = *(eiPlus1+i*N+j) - *(ei+i*N+j);
*errorIplus1 += diff*diff;
}
}
numIter++;
if (*errorIplus1<=0.00001) {
break;
}
}
float sigmaArr[M*N];
float* sigmaPtr = sigmaArr;
float sigmaInv_arr[N*M];
float v_arr[N*N];
float* sigmaInv = sigmaInv_arr;
float* v = v_arr;
for (int i=0; i<N; i++) {
int newIndex = 0;
float current = fabs(*(diPlus1+i*N+i));
for (int j=0; j<N; j++) {
float other = fabs(*(diPlus1+j*N+j));
if (other>current) {
newIndex++;
} else if (other==current) {
if (i<j) {
newIndex++;
}
}
}
for (int j=0; j<N; j++) {
*(*(V_T)+newIndex*N+j) = *(eiPlus1+j*N+newIndex);
*(v+j*N+newIndex) = *(eiPlus1+j*N+newIndex);
if (j==newIndex) {
*(sigmaInv+newIndex*M+newIndex) = 1.0/sqrt(current);
*(sigmaPtr+newIndex*N+newIndex) = sqrt(current);
*(*(SIGMA)+newIndex) = sqrt(current);
} else {
*(sigmaInv+j*M+newIndex) = 0.0;
*(sigmaPtr+j*N+newIndex) = 0.0;
}
}
}
for (int i=0; i<N; i++) {
for (int j=N; j<M; j++) {
*(sigmaInv+i*M+j) = 0.0;
*(sigmaPtr+j*N+i) = 0.0;
}
}
float mv_arr[M*N];
float* mv = mv_arr;
matmul(D, v, mv, M, N, N);
matmul(mv, sigmaInv, *U, M, N, M);
}
void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T) {
float* U1 = (float*) malloc(sizeof(float)*M*M);
float* VT1 = (float*) malloc(sizeof(float)*N*N);
SVD_internal(M, N, D, &U1, SIGMA, &VT1);
transpose(*U, VT1, N, N);
transpose(*V_T, U1, M, M);
free(U1);
free(VT1);
}
void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
float totalEigenSum = 0.0;
for (int i=0; i<N; i++) {
float val = *(SIGMA+i);
totalEigenSum += val*val;
}
float eigenSum = 0.0;
for (*K=0; *K<N; (*K)++) {
float val = *(SIGMA+(*K));
eigenSum += val*val;
float percent = eigenSum/totalEigenSum;
if (percent >= retention/100.0) {
(*K)++;
break;
}
}
float* D_HAT_TEMP = (float*) malloc(sizeof(float)*M*(*K));
for (int i=0; i<M; i++) {
for (int j=0; j<(*K); j++) {
*(D_HAT_TEMP+i*(*K)+j) = 0.0;
for (int k=0; k<N; k++) {
*(D_HAT_TEMP+i*(*K)+j) += (*(D+i*N+k))*(*(U+k*N+j));
}
}
}
*D_HAT = D_HAT_TEMP;
}
