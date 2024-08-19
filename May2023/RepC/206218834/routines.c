#include "matrix.h"
void scalar_multiply(struct COO matrix, double scalar) {
int i;
if (matrix.type == TYPE_INT) {
#pragma omp parallel for shared(matrix,scalar) num_threads(param.threads)
for (i = 0; i < matrix.count; i++) {
matrix.elements[i].value.f = (double) matrix.elements[i].value.i * scalar;
}
} else {
#pragma omp parallel for shared(matrix,scalar) num_threads(param.threads)
for (i = 0; i < matrix.count; i++) {
matrix.elements[i].value.f *= scalar;
}
}
}
int trace(struct CSR matrix) {
int trace = 0;
int i;
#pragma omp parallel for reduction(+:trace) shared(matrix) num_threads(param.threads)
for (i = 1; i < matrix.rows + 1; i++) {
int elements = matrix.ia[i] - matrix.ia[i-1];
if (elements == 0) {
continue;
}
int pos = matrix.ia[i-1]; 
while (pos - matrix.ia[i-1] <= elements) {
if (matrix.ja[pos] == i - 1) {
trace += matrix.nnz.i[pos];
break;
} else if (matrix.ja[pos] > i - 1) {
break;
}
pos++;
}
}
return trace;
}
double trace_f(struct CSR matrix) {
double trace = 0.0;
int i;
#pragma omp parallel for reduction(+:trace) shared(matrix) num_threads(param.threads)
for (i = 1; i < matrix.rows + 1; i++) {
int elements = matrix.ia[i] - matrix.ia[i-1];
if (elements == 0) {
continue;
}
int pos = matrix.ia[i-1]; 
while (pos - matrix.ia[i-1] <= elements) {
if (matrix.ja[pos] == i - 1) {
trace += matrix.nnz.f[pos];
break;
} else if (matrix.ja[pos] > i - 1) {
break;
}
pos++;
}
}
return trace;
}
struct COO matrix_addition(struct CSR matrix, struct CSR matrix2) {
int totalcount = 0;
int i;
struct COO *result_local = allocate(matrix.rows * sizeof(struct COO));
for (i = 0; i < matrix.rows; i++) {
result_local[i].count = 0;
result_local[i].elements = allocate((matrix.count + matrix2.count) * sizeof(struct ELEMENT));
}
#pragma omp parallel for reduction(+:totalcount) shared(matrix,matrix2) num_threads(param.threads)
for (i = 0; i < matrix.rows; i++) {
int elements = matrix.ia[i+1] - matrix.ia[i];
int elements2 = matrix2.ia[i+1] - matrix2.ia[i];
int m1seen = 0;
int m2seen = 0;
int pos = matrix.ia[i];
int pos2 = matrix2.ia[i];
while (m1seen != elements || m2seen != elements2) {
if (m1seen == elements) { 
while (m2seen < elements2) {
result_local[i].elements[result_local[i].count].y = matrix2.ja[pos2];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.i = matrix2.nnz.i[pos2++];
m2seen++;
}
break;
} else if (m2seen == elements2) { 
while (m1seen < elements) {
result_local[i].elements[result_local[i].count].y = matrix.ja[pos];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.i = matrix.nnz.i[pos++];
m1seen++;
}
break;
} else { 
if (matrix.ja[pos] < matrix2.ja[pos2]) { 
result_local[i].elements[result_local[i].count].y = matrix.ja[pos];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.i = matrix.nnz.i[pos++]; 
m1seen++;
} else if (matrix2.ja[pos2] < matrix.ja[pos]) { 
result_local[i].elements[result_local[i].count].y = matrix2.ja[pos2];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.i = matrix2.nnz.i[pos2++]; 
m2seen++;
} else { 
result_local[i].elements[result_local[i].count].y = matrix2.ja[pos2];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.i = matrix.nnz.i[pos++] + matrix2.nnz.i[pos2++]; 
m1seen++;
m2seen++;
}
}
}
totalcount += result_local[i].count;
}
struct COO result;
result.rows = matrix.rows;
result.cols = matrix.cols;
result.type = matrix.type;
result.elements = allocate(totalcount * sizeof(struct ELEMENT));
result.count = 0;
for (i = 0; i < matrix.rows; i++) {
if (result_local[i].count != 0) {
memcpy(&result.elements[result.count], result_local[i].elements, result_local[i].count * sizeof(struct ELEMENT));
result.count += result_local[i].count;
}
free(result_local[i].elements);
}
free(result_local);
return result;
}
struct COO matrix_addition_f(struct CSR matrix, struct CSR matrix2) {
int totalcount = 0;
int i;
struct COO *result_local = allocate(matrix.rows * sizeof(struct COO));
for (i = 0; i < matrix.rows; i++) {
result_local[i].count = 0;
result_local[i].elements = allocate((matrix.count + matrix2.count) * sizeof(struct ELEMENT));
}
#pragma omp parallel for reduction(+:totalcount) shared(matrix,matrix2) num_threads(param.threads)
for (i = 0; i < matrix.rows; i++) {
int elements = matrix.ia[i+1] - matrix.ia[i];
int elements2 = matrix2.ia[i+1] - matrix2.ia[i];
int m1seen = 0;
int m2seen = 0;
int pos = matrix.ia[i];
int pos2 = matrix2.ia[i];
while (m1seen != elements || m2seen != elements2) {
if (m1seen == elements) { 
while (m2seen < elements2) {
result_local[i].elements[result_local[i].count].y = matrix2.ja[pos2];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.f = matrix2.nnz.f[pos2++];
m2seen++;
}
break;
} else if (m2seen == elements2) { 
while (m1seen < elements) {
result_local[i].elements[result_local[i].count].y = matrix.ja[pos];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.f = matrix.nnz.f[pos++];
m1seen++;
}
break;
} else { 
if (matrix.ja[pos] < matrix2.ja[pos2]) { 
result_local[i].elements[result_local[i].count].y = matrix.ja[pos];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.f = matrix.nnz.f[pos++]; 
m1seen++;
} else if (matrix2.ja[pos2] < matrix.ja[pos]) { 
result_local[i].elements[result_local[i].count].y = matrix2.ja[pos2];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.f = matrix2.nnz.f[pos2++]; 
m2seen++;
} else { 
result_local[i].elements[result_local[i].count].y = matrix2.ja[pos2];
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].value.f = matrix.nnz.f[pos++] + matrix2.nnz.f[pos2++]; 
m1seen++;
m2seen++;
}
}
}
totalcount += result_local[i].count;
}
struct COO result;
result.rows = matrix.rows;
result.cols = matrix.cols;
result.type = matrix.type;
result.elements = allocate(totalcount * sizeof(struct ELEMENT));
result.count = 0;
for (i = 0; i < matrix.rows; i++) {
if (result_local[i].count != 0) {
memcpy(&result.elements[result.count], result_local[i].elements, result_local[i].count * sizeof(struct ELEMENT));
result.count += result_local[i].count;
}
free(result_local[i].elements);
}
free(result_local);
return result;
}
struct CSR transpose(struct CSC matrix) {
struct CSR result;
int i;
result.rows = matrix.cols;
result.cols = matrix.rows;
result.ia = matrix.ia;
result.ja = matrix.ja;
result.type = matrix.type;
if (matrix.type == TYPE_INT) {
result.nnz.i = allocate(matrix.count * sizeof(int));
#pragma omp parallel for shared(result,matrix) num_threads(param.threads)
for (i = 0; i < matrix.count; i++) {
result.nnz.i[i] = matrix.nnz.i[i];
}
} else {
result.nnz.f = allocate(matrix.count * sizeof(double));
#pragma omp parallel for shared(result,matrix) num_threads(param.threads)
for (i = 0; i < matrix.count; i++) {
result.nnz.f[i] = matrix.nnz.f[i];
}
}
return result;
}
struct COO matrix_multiply(struct CSR matrix, struct CSC matrix2) {
int totalcount = 0;
int i;
struct COO *result_local = allocate(matrix.rows * sizeof(struct COO));
for (i = 0; i < matrix.rows; i++) {
result_local[i].count = 0;
result_local[i].elements = allocate((matrix2.cols) * sizeof(struct ELEMENT));
}
#pragma omp parallel for reduction(+:totalcount) shared(matrix,matrix2) num_threads(param.threads)
for (i = 0; i < matrix.rows; i++) {
int m1count = matrix.ia[i+1] - matrix.ia[i];
for (int j = 0; j < matrix2.cols; j++) {
int dp = 0; 
int m1pos = matrix.ia[i];
int m1seen = 0;
int m2count = matrix2.ia[j+1] - matrix2.ia[j];
int m2pos = matrix2.ia[j];
int m2seen = 0;
while (m1seen != m1count && m2seen != m2count) {
if (matrix.ja[m1pos] == matrix2.ja[m2pos]) { 
dp += (matrix.nnz.i[m1pos++] * matrix2.nnz.i[m2pos++]);
m1seen++;
m2seen++;
} else if (matrix.ja[m1pos] < matrix2.ja[m2pos]) {
m1seen++;
m1pos++;
} else {
m2seen++;
m2pos++;
}
}
if (dp != 0) {
result_local[i].elements[result_local[i].count].value.i = dp;
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].y = j;
}
}
totalcount += result_local[i].count;
}
struct COO result;
result.count = 0;
result.type = matrix.type;
result.elements = allocate(totalcount * sizeof(struct ELEMENT));
result.rows = matrix.rows;
result.cols = matrix2.cols;
for (i = 0; i < matrix.rows; i++) {
if (result_local[i].count != 0) {
memcpy(&result.elements[result.count], result_local[i].elements, result_local[i].count * sizeof(struct ELEMENT));
result.count += result_local[i].count;
}
free(result_local[i].elements);
}
free(result_local);
return result;
}
struct COO matrix_multiply_f(struct CSR matrix, struct CSC matrix2) {
int totalcount = 0;
int i;
struct COO *result_local = allocate(matrix.rows * sizeof(struct COO));
for (i = 0; i < matrix.rows; i++) {
result_local[i].count = 0;
result_local[i].elements = allocate((matrix2.cols) * sizeof(struct ELEMENT));
}
#pragma omp parallel for reduction(+:totalcount) shared(matrix,matrix2) num_threads(param.threads)
for (i = 0; i < matrix.rows; i++) {
int m1count = matrix.ia[i+1] - matrix.ia[i];
for (int j = 0; j < matrix2.cols; j++) {
double dp = 0; 
int m1pos = matrix.ia[i];
int m1seen = 0;
int m2count = matrix2.ia[j+1] - matrix2.ia[j];
int m2pos = matrix2.ia[j];
int m2seen = 0;
while (m1seen != m1count && m2seen != m2count) {
if (matrix.ja[m1pos] == matrix2.ja[m2pos]) { 
dp += (matrix.nnz.f[m1pos++] * matrix2.nnz.f[m2pos++]);
m1seen++;
m2seen++;
} else if (matrix.ja[m1pos] < matrix2.ja[m2pos]) {
m1seen++;
m1pos++;
} else {
m2seen++;
m2pos++;
}
}
if (dp != 0) {
result_local[i].elements[result_local[i].count].value.f = dp;
result_local[i].elements[result_local[i].count].x = i;
result_local[i].elements[result_local[i].count++].y = j;
}
}
totalcount += result_local[i].count;
}
struct COO result;
result.count = 0;
result.type = matrix.type;
result.elements = allocate(totalcount * sizeof(struct ELEMENT));
result.rows = matrix.rows;
result.cols = matrix2.cols;
for (i = 0; i < matrix.rows; i++) {
if (result_local[i].count != 0) {
memcpy(&result.elements[result.count], result_local[i].elements, result_local[i].count * sizeof(struct ELEMENT));
result.count += result_local[i].count;
}
free(result_local[i].elements);
}
free(result_local);
return result;
}
