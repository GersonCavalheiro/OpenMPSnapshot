#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <ctime>

#define THREADS omp_get_num_procs()

#define SIZE 2000 
#define BORDER 64 

template <typename T>
T** new_matrix(int height, int width) {
T** matrix = new T * [height];
for (int i = 0; i < height; ++i) {
matrix[i] = new T[width];
}
return matrix;
}

template <typename T>
void delete_matrix(T** matrix, int height) {
for (int i = 0; i < height; ++i) {
delete[] matrix[i];
}
delete[] matrix;
}

void generate_matrix(int** matrix, int height, int width, bool zero) {
if (zero == true) {
for (int i = 0; i < height; ++i)
for (int j = 0; j < width; ++j) matrix[i][j] = 0;
}
else {
for (int i = 0; i < height; ++i)
for (int j = 0; j < width; ++j) matrix[i][j] = (i + j) % 100;
}
}

void print_matrix(int size, int** M) {
for (int ix = 0; ix < size; ++ix) {
for (int iy = 0; iy < size; ++iy) {
std::cout << M[ix][iy] << " ";
}
std::cout << std::endl;
}
std::cout << std::endl;
}

void split_matrix(int A_size, int** A, int** A_11, int** A_12, int** A_21, int** A_22) {

int next_size = (A_size / 2);
if (A_size % 2 == 1) { next_size++; }

for (int i = 0; i < A_size / 2; ++i) {
std::copy(A[i], A[i] + next_size, A_11[i]);
std::copy(A[i] + next_size, A[i] + A_size, A_12[i]);
std::copy(A[i + next_size], A[i + next_size] + next_size, A_21[i]);
std::copy(A[i + next_size] + next_size, A[i  + next_size] + A_size, A_22[i]);
}
if (A_size % 2 == 1) {
std::copy(A[next_size - 1], A[next_size - 1] + next_size, A_11[next_size - 1]);
std::copy(A[next_size - 1] + next_size, A[next_size - 1] + A_size, A_12[next_size - 1]);
for (int i = 0; i < next_size; ++i) {
A_12[i][next_size - 1] = 0;
A_21[next_size - 1][i] = 0;
A_22[next_size - 1][i] = 0;
A_22[i][next_size - 1] = 0;
}
}
}

void collect_matrix(int A_size, int** A, int** A_11, int** A_12, int** A_21, int** A_22) {
int next_size = (A_size / 2);
if (A_size % 2 == 1) { next_size++; }

for (int i = 0; i < A_size / 2; ++i) {
std::copy(A_11[i], A_11[i] + next_size, A[i]);
std::copy(A_12[i], A_12[i] + A_size - next_size, A[i] + next_size);
std::copy(A_21[i], A_21[i] + next_size, A[i + next_size]);
std::copy(A_22[i], A_22[i] + A_size - next_size, A[i + next_size] + next_size);
}	
if (A_size % 2 == 1) {
std::copy(A_11[next_size - 1], A_11[next_size - 1] + next_size, A[next_size - 1]);
std::copy(A_12[next_size - 1], A_12[next_size - 1] + A_size - next_size, A[next_size - 1] + next_size);
}
}

int** matrix_summation(int size, int** A, int** B, int** C, bool minus) {
if (minus == false) {
for (int ix = 0; ix < size; ++ix) {
#pragma omp simd
for (int iy = 0; iy < size; ++iy) {
C[ix][iy] = A[ix][iy] + B[ix][iy];
}
}
}
else {
for (int ix = 0; ix < size; ++ix) {
#pragma omp simd
for (int iy = 0; iy < size; ++iy) {
C[ix][iy] = A[ix][iy] - B[ix][iy];
}
}
}
return C;
}



void matrix_multiply(int size, int** A, int** B, int** C) {

int** B_trans = new_matrix<int>(size, size);
for (int ix = 0; ix < size; ++ix) {
#pragma omp simd
for (int iy = 0; iy < size; ++iy) {
B_trans[iy][ix] = B[ix][iy];
}
}

for (int ix = 0; ix < size; ++ix) {
for (int iy = 0; iy < size; ++iy) {
#pragma omp simd
for (int i = 0; i < size; ++i) {
C[ix][iy] += A[ix][i] * B_trans[iy][i];
}
}
}
}

int** Strassen_multiply(int size, int** A, int** B, int** C) {

if (size < BORDER) {
matrix_multiply(size, A, B, C);
}
else {
int next_size = (size / 2);
if (size % 2 == 1) { next_size++; }

int** A_11 = new_matrix<int>(next_size, next_size);
int** A_12 = new_matrix<int>(next_size, next_size);
int** A_21 = new_matrix<int>(next_size, next_size);
int** A_22 = new_matrix<int>(next_size, next_size);

int** B_11 = new_matrix<int>(next_size, next_size);
int** B_12 = new_matrix<int>(next_size, next_size);
int** B_21 = new_matrix<int>(next_size, next_size);
int** B_22 = new_matrix<int>(next_size, next_size);

int** C_11 = new_matrix<int>(next_size, next_size);
int** C_12 = new_matrix<int>(next_size, next_size);
int** C_21 = new_matrix<int>(next_size, next_size);
int** C_22 = new_matrix<int>(next_size, next_size);

generate_matrix(C_11, next_size, next_size, true);
generate_matrix(C_12, next_size, next_size, true);
generate_matrix(C_21, next_size, next_size, true);
generate_matrix(C_22, next_size, next_size, true);

split_matrix(size, A, A_11, A_12, A_21, A_22);
split_matrix(size, B, B_11, B_12, B_21, B_22);

int** P_1 = new_matrix<int>(next_size, next_size);
int** P_2 = new_matrix<int>(next_size, next_size);
int** P_3 = new_matrix<int>(next_size, next_size);
int** P_4 = new_matrix<int>(next_size, next_size);
int** P_5 = new_matrix<int>(next_size, next_size);
int** P_6 = new_matrix<int>(next_size, next_size);
int** P_7 = new_matrix<int>(next_size, next_size);


#pragma omp task shared(P_1)
{	
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_1, next_size, next_size, true);
P_1 = Strassen_multiply(next_size, matrix_summation(next_size, A_11, A_22, Buff_1, false), matrix_summation(next_size, B_11, B_22, Buff_2, false), P_1); 
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}
#pragma omp task  shared(P_2)
{
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_2, next_size, next_size, true);
P_2 = Strassen_multiply(next_size, matrix_summation(next_size, A_21, A_22, Buff_1, false), B_11, P_2);
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}
#pragma omp task  shared(P_3)
{
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_3, next_size, next_size, true);
P_3 = Strassen_multiply(next_size, A_11, matrix_summation(next_size, B_12, B_22, Buff_1, true), P_3);
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}
#pragma omp task  shared(P_4)
{
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_4, next_size, next_size, true);
P_4 = Strassen_multiply(next_size, A_22, matrix_summation(next_size, B_21, B_11, Buff_1, true), P_4);
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}
#pragma omp task  shared(P_5)
{
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_5, next_size, next_size, true);
P_5 = Strassen_multiply(next_size, matrix_summation(next_size, A_11, A_12, Buff_1, false), B_22, P_5);
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}
#pragma omp task  shared(P_6)
{
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_6, next_size, next_size, true);
P_6 = Strassen_multiply(next_size, matrix_summation(next_size, A_21, A_11, Buff_1, true), matrix_summation(next_size, B_11, B_12, Buff_2, false), P_6);
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}
#pragma omp task  shared(P_7)
{
int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);
generate_matrix(P_7, next_size, next_size, true);
P_7 = Strassen_multiply(next_size, matrix_summation(next_size, A_12, A_22, Buff_1, true), matrix_summation(next_size, B_21, B_22, Buff_2, false), P_7);
delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);
}

#pragma omp taskwait

delete_matrix<int>(A_11, next_size);
delete_matrix<int>(A_12, next_size);
delete_matrix<int>(A_21, next_size);
delete_matrix<int>(A_22, next_size);

delete_matrix<int>(B_11, next_size);
delete_matrix<int>(B_12, next_size);
delete_matrix<int>(B_21, next_size);
delete_matrix<int>(B_22, next_size);

int** Buff_1 = new_matrix<int>(next_size, next_size);
int** Buff_2 = new_matrix<int>(next_size, next_size);

C_11 = matrix_summation(next_size, matrix_summation(next_size, P_1, P_7, Buff_1, false), matrix_summation(next_size, P_4, P_5, Buff_2, true), C_11, false);
C_12 = matrix_summation(next_size, P_3, P_5, C_12, false);
C_21 = matrix_summation(next_size, P_2, P_4, C_21, false);
C_22 = matrix_summation(next_size, matrix_summation(next_size, P_1, P_2, Buff_1, true), matrix_summation(next_size, P_3, P_6, Buff_2, false), C_22, false);

delete_matrix<int>(Buff_1, next_size);
delete_matrix<int>(Buff_2, next_size);

delete_matrix<int>(P_1, next_size);
delete_matrix<int>(P_2, next_size);
delete_matrix<int>(P_3, next_size);
delete_matrix<int>(P_4, next_size);
delete_matrix<int>(P_5, next_size);
delete_matrix<int>(P_6, next_size);
delete_matrix<int>(P_7, next_size);

collect_matrix(size, C, C_11, C_12, C_21, C_22);

delete_matrix<int>(C_11, next_size);
delete_matrix<int>(C_12, next_size);
delete_matrix<int>(C_21, next_size);
delete_matrix<int>(C_22, next_size);
}
return C;
}


int main() {

int** matrix_A = new_matrix<int>(SIZE, SIZE);
int** matrix_B = new_matrix<int>(SIZE, SIZE);
int** matrix_C = new_matrix<int>(SIZE, SIZE);

generate_matrix(matrix_A, SIZE, SIZE, false);
generate_matrix(matrix_B, SIZE, SIZE, false);
generate_matrix(matrix_C, SIZE, SIZE, true);

double time_d = 0;
double time_end = 0;
double time_start = 0;

std::cout << "Calculations begin, cores: " << THREADS << std::endl;
time_start = omp_get_wtime();

#pragma omp parallel num_threads(THREADS)
{
#pragma omp single 
{
Strassen_multiply(SIZE, matrix_A, matrix_B, matrix_C);
}
}

time_end = omp_get_wtime();
time_d = time_end - time_start;
std::cout << "Time: " << double(time_d) << " seconds" << std::endl;


delete_matrix<int>(matrix_A, SIZE);
delete_matrix<int>(matrix_B, SIZE);
delete_matrix<int>(matrix_C, SIZE);

return 0;
}

