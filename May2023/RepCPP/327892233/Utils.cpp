#include <string>
#include <mkl.h>
#include <omp.h>

class Utils {
public:
void printMatrix(double* mat, int m, int n) {
printf("\t\t");
for (int i = 0; i < m * n; i++)
{
printf("%4.4f\t", mat[i]);
if ((i + 1) % n == 0) {
printf("\n\t\t");
}
}
printf("\n");
}


double* clone(double* matrix, int size) {
double* res = new double[size];
memcpy(res, matrix, sizeof(double) * size);
return res;
}

double * centerData(double * data, int m, int n) {
double* means = new double[n];

#pragma omp parallel num_threads(11)
{
int id = omp_get_thread_num();

#pragma omp sections nowait
{
#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n];
}
means[0] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n] -= means[0];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 1];
}
means[1] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 1] -= means[1];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 2];
}
means[2] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 2] -= means[2];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 3];
}
means[3] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 3] -= means[3];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 4];
}
means[4] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 4] -= means[4];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 5];
}
means[5] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 5] -= means[5];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 6];
}
means[6] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 6] -= means[6];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 7];
}
means[7] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 7] -= means[7];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 8];
}
means[8] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 8] -= means[8];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 9];
}
means[9] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 9] -= means[9];
}
}

#pragma omp section
{
double sum = 0;
for (int i = 0; i < m; i++) {
sum += data[i * n + 10];
}
means[10] = sum / m;

for (int i = 0; i < m; i++) {
data[i * n + 10] -= means[10];
}
}
}
}
return data;
}

double * computeCov(double * data, int m_, int n_) {
MKL_INT m, k, n, alpha, beta, lda, ldb, ldc;
CBLAS_LAYOUT layout;
CBLAS_TRANSPOSE transA;
CBLAS_TRANSPOSE transB;
layout = CblasRowMajor;
transA = CblasTrans;
transB = CblasNoTrans;
m = n_; 
n = n_; 
k = m_; 
alpha = 1;
beta = 0;
lda = m;
ldb = n;
ldc = m;

double* data_Trans = clone(data, m_ * n_);
double * Z = new double[n_ * n_]{ 0.0 }; 

cblas_dgemm(layout, transA, transB, m, n, k, alpha, data_Trans, lda, data, ldb, beta, Z, ldc);

#pragma omp parallel num_threads(8)
{
#pragma omp for nowait
for (int i = 0; i < n_ * n_; i++) {
Z[i] /= m_;
}
}

return Z;
}
};