#include <iostream>
#include <string>
#include <array>
#include <random>
#include <bits/stdc++.h>
#include <omp.h>
#include <stdexcept>


std::mt19937 rng;


template<typename T, std::size_t rows, std::size_t columns>
using matrix = std::array<std::array<T, columns>, rows>;


int randint(int from, int to) {
return std::uniform_int_distribution<int>(from, to)(rng);
}


double randdouble(double from, double to) {
return std::uniform_real_distribution<double>(from, to)(rng);
}



inline void transpose_block_SSE4x4(float *A, float *B, const int n, const int m, const int lda, const int ldb ,const int block_size) {
#pragma omp parallel for
for(int i=0; i<n; i+=block_size) {
for(int j=0; j<m; j+=block_size) {
int max_i2 = i+block_size < n ? i + block_size : n;
int max_j2 = j+block_size < m ? j + block_size : m;
for(int i2=i; i2<max_i2; i2+=4) {
for(int j2=j; j2<max_j2; j2+=4) {
}
}
}
}
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> add_matrices \
(matrix<double, rows, columns> mat1, matrix<double, rows, columns> mat2) {
matrix<double, rows, columns> result_matrix;
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
result_matrix[i][j] = mat1[i][j] + mat2[i][j];
}
}
return result_matrix;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> set_above_main_diagonal \
(const matrix<double, rows, columns> &mat, double num) {
matrix<double, rows, columns> new_mat = mat;
for (int i = 0; i < rows; ++i) {
for (int j = i+1; j < columns; ++j) {
new_mat[i][j] = num;
}
}
return new_mat;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> set_above_sec_diagonal \
(const matrix<double, rows, columns> &mat, double num) {
matrix<double, rows, columns> new_mat = mat;
for (int i = 0; i < rows; ++i) {
for (int j = columns-i-2; j >= 0; --j) {
new_mat[i][j] = num;
}
}
return new_mat;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> set_under_main_diagonal \
(const matrix<double, rows, columns> &mat, const double num) {
matrix<double, rows, columns> new_mat = mat;
for (int i = rows-1; i > 0; --i) {
for (int j = i-1; j >= 0; --j) {
new_mat[i][j] = num;
}
}
return new_mat;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> set_under_sec_diagonal \
(const matrix<double, rows, columns> &mat, const double num) {
int col = (int)columns;
matrix<double, rows, columns> new_mat = mat;
for (int i = 1; i < rows; ++i) {
for (int j = std::clamp(col-i, 0, col-1); j < columns; ++j) {
new_mat[i][j] = num;
}
}
return new_mat;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> set_main_diagonal \
(const matrix<double, rows, columns> &mat, const double num) {
matrix<double, rows, columns> new_mat = mat;
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
new_mat[i][j] = i == j ? num : new_mat[i][j];
}
}
return new_mat;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> set_sec_diagonal \
(const matrix<double, rows, columns> &mat, const double num) {
matrix<double, rows, columns> new_mat = mat;
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
new_mat[i][j] = i == columns-1-j ? num : new_mat[i][j];
}
}
return new_mat;
}


template<std::size_t rows, std::size_t columns>
void show_mat(const matrix<double, rows, columns> mat) {
for (std::array<double, columns> row : mat) {
for (double item : row) {
std::cout << item << "\t";
}
std::cout << "\n";
}
std::cout << std::endl;
}


template<std::size_t rows, std::size_t columns>
void fill_mat_(matrix<double, rows, columns> &mat, double n = -1337.) {
if (n == -1337.) {
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
mat[i][j] = randdouble(0, 9);
}
}
} else {
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
mat[i][j] = n;
}
}
}
}


template<std::size_t rows, std::size_t columns>
void fill_identity_mat_(matrix<double, rows, columns> &mat) {
if (rows != columns) {
throw std::invalid_argument \
("Row number must equal col number for identity matrices");
}

for (int i = 0; i < rows; ++i) {
for (int j = 0; j < rows; ++j) {
mat[i][j] = i == j ? 1 : 0;
}
}
}


template<std::size_t rows1, std::size_t columns1, \
std::size_t rows2, std::size_t columns2>
matrix<double, rows1, columns2> multiply_matrices \
(matrix<double, rows1, columns1> mat1, matrix<double, rows2, columns2> mat2) {
assert(rows2 == columns1);
matrix<double, rows1, columns2> result_matrix;
# pragma omp parallel shared ( mat1, mat2, result_matrix, \
rows1, columns1, rows2, columns2 ) private ( i, j, k )
{
fill_mat_(result_matrix, 0);
# pragma omp for
for(int i = 0; i < rows1; ++i) {
for(int j = 0; j < columns2; ++j) {
for(int k = 0; k < columns1; ++k) {
result_matrix[i][j] += mat1[i][k] * mat2[k][j];
}
}
}
}
return result_matrix;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> multiply_matrix \
(matrix<double, rows, columns> mat, double num) {
matrix<double, rows, columns> result_matrix;

for (int i = 0; i < rows; ++i) {
for (int j = 0; j < columns; ++j) {
result_matrix[i][j] = mat[i][j]*num;
}
}

return result_matrix;
}


template<std::size_t rows, std::size_t columns>
matrix<double, rows, columns> matrix_pow \
(matrix<double, rows, columns> mat, int num) {
if (num < 1) {
throw std::invalid_argument("A power cannot be less than 1");
}
--(--num);
matrix<double, rows, columns> result_matrix = \
multiply_matrices(mat, mat);

for (int i = 0; i < num; ++i) {
show_mat(result_matrix);
result_matrix = multiply_matrices(result_matrix, mat);
}
return result_matrix;
}


template<std::size_t rows, std::size_t columns>
int matrix_det(matrix<double, rows, columns> mat) {
if (rows != columns) {
throw std::invalid_argument("Only a square matrices have a det");
}

double tmp, det;

matrix<double, rows, columns> tmp_mat = mat;
for (int k = 0; k < rows - 1; ++k) {
for (int i = k + 1; i < rows; ++i) {
tmp = -tmp_mat[i][k] / tmp_mat[k][k];
for (int j = 0; j < rows; ++j) {
tmp_mat[i][j] += tmp_mat[k][j] * tmp;
}
}
}

det = 1;
for (int i = 0; i < rows; ++i) {
det *= tmp_mat[i][i];
}

return det;
}


template<std::size_t rows, std::size_t columns>
int matrix_rank(matrix<double, rows, columns> A) {
const double EPS = 1E-9;
int n = A.size();
int m = A[0].size();

int rank = 0;
std::vector<bool> row_selected(n, false);
for (int i = 0; i < m; ++i) {
int j;
for (j = 0; j < n; ++j) {
if (!row_selected[j] && abs(A[j][i]) > EPS)
break;
}

if (j != n) {
++rank;
row_selected[j] = true;
for (int p = i + 1; p < m; ++p)
A[j][p] /= A[j][i];
for (int k = 0; k < n; ++k) {
if (k != j && abs(A[k][i]) > EPS) {
for (int p = i + 1; p < m; ++p)
A[k][p] -= A[j][p] * A[k][i];
}
}
}
}
return rank;
}


void assert_test() {
const matrix<double, 3, 3> square3x3 = \
{{{1,2,3}, {4,5,6}, {7,8,9}}};
const matrix<double, 4, 4> square4x4 = \
{{{1,2,3,4}, {1,2,3,4}, {1,2,3,4}, {1,2,3,4}}};
const matrix<double, 5, 3> ver_rect = \
{{{1,2,3}, {1,2,3}, {1,2,3}, {1,2,3}, {1,2,3}}};
const matrix<double, 3, 5> hor_rect = \
{{{1,2,3,4,5}, {1,2,3,4,5}, {1,2,3,4,5}}};

const matrix<double, 3, 5> mult1 = \
{{{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}};
const matrix<double, 5, 3> mult2 = \
{{{4, 4, 4}, {4, 4, 4}, {4, 4, 4}, {4, 4, 4}, {4, 4, 4}}};

const matrix<double, 3, 3> sum = \
{{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};

matrix<double, 3, 3> filler_mat = \
{{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}};

matrix<double, 3, 3> imat;

const matrix<double, 3, 3> imat_ref = \
{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

const matrix<double, 3, 3> res_above_square3x3 = \
{{{1,0,0}, {4,5,0}, {7,8,9}}};
const matrix<double, 4, 4> res_above_square4x4 = \
{{{1,0,0,0}, {1,2,0,0}, {1,2,3,0}, {1,2,3,4}}};
const matrix<double, 5, 3> res_above_ver_rect = \
{{{1,0,0}, {1,2,0}, {1,2,3}, {1,2,3}, {1,2,3}}};
const matrix<double, 3, 5> res_above_hor_rect = \
{{{1,0,0,0,0}, {1,2,0,0,0}, {1,2,3,0,0}}};

const matrix<double, 3, 3> res_main_square3x3 = \
{{{0,2,3}, {4,0,6}, {7,8,0}}};
const matrix<double, 4, 4> res_main_square4x4 = \
{{{0,2,3,4}, {1,0,3,4}, {1,2,0,4}, {1,2,3,0}}};
const matrix<double, 5, 3> res_main_ver_rect = \
{{{0,2,3}, {1,0,3}, {1,2,0}, {1,2,3}, {1,2,3}}};
const matrix<double, 3, 5> res_main_hor_rect = \
{{{0,2,3,4,5}, {1,0,3,4,5}, {1,2,0,4,5}}};

const matrix<double, 3, 3> res_under_square3x3 = \
{{{1,2,3}, {0,5,6}, {0,0,9}}};
const matrix<double, 4, 4> res_under_square4x4 = \
{{{1,2,3,4}, {0,2,3,4}, {0,0,3,4}, {0,0,0,4}}};
const matrix<double, 5, 3> res_under_ver_rect = \
{{{1,2,3}, {0,2,3}, {0,0,3}, {0,0,0}, {0,0,0}}};
const matrix<double, 3, 5> res_under_hor_rect = \
{{{1,2,3,4,5}, {0,2,3,4,5}, {0,0,3,4,5}}};

const matrix<double, 3, 3> res_mult = \
{{{40, 40, 40}, {40, 40, 40,}, {40, 40, 40}}};

const matrix<double, 3, 3> res_mult_num = \
{{{3, 3, 3}, {3, 3, 3}, {3, 3, 3}}};

const matrix<double, 3, 3> res_fill = \
{{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};

const matrix<double, 3, 3> res_sum = \
{{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}};

const matrix<double, 3, 3> res_pow = \
{{{468, 576, 684}, {1062, 1305, 1548}, {1656, 2034, 2412}}};

const matrix<double, 5, 3> res_transpose = \
{{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}}};

const matrix<double, 3, 3> res_aux = \
{{{1, 2, 0}, {4, 0, 6}, {0, 8, 9}}};
const matrix<double, 3, 3> res_above_aux = \
{{{0, 0, 3}, {0, 5, 6}, {7, 8, 9}}};
const matrix<double, 3, 3> res_under_aux = \
{{{1, 2, 3}, {4, 5, 0}, {7, 0, 0}}};

fill_mat_(filler_mat, 1);
fill_identity_mat_(imat);

assert(set_above_main_diagonal(square3x3, 0) == res_above_square3x3);
assert(set_above_main_diagonal(square4x4, 0) == res_above_square4x4);
assert(set_above_main_diagonal(hor_rect, 0) == res_above_hor_rect);
assert(set_above_main_diagonal(ver_rect, 0) == res_above_ver_rect);

assert(set_main_diagonal(square3x3, 0) == res_main_square3x3);
assert(set_main_diagonal(square4x4, 0) == res_main_square4x4);
assert(set_main_diagonal(hor_rect, 0) == res_main_hor_rect);
assert(set_main_diagonal(ver_rect, 0) == res_main_ver_rect);

assert(set_under_main_diagonal(square3x3, 0) == res_under_square3x3);
assert(set_under_main_diagonal(square4x4, 0) == res_under_square4x4);
assert(set_under_main_diagonal(hor_rect, 0) == res_under_hor_rect);
assert(set_under_main_diagonal(ver_rect, 0) == res_under_ver_rect);

assert(multiply_matrix(sum, 3) == res_mult_num);
assert(multiply_matrices(mult1, mult2) == res_mult);
assert(add_matrices(sum, sum) == res_sum);
assert(matrix_pow(square3x3, 3) == res_pow);

assert(filler_mat == res_fill);
assert(imat == imat_ref);
assert(square3x3 == multiply_matrices(square3x3, imat));

assert(set_sec_diagonal(square3x3, 0) == res_aux);
assert(set_above_sec_diagonal(square3x3, 0) == res_above_aux);
assert(set_under_sec_diagonal(square3x3, 0) == res_under_aux);

assert(matrix_det(square3x3) == 0);
assert(matrix_rank(square3x3) == 2);
assert(matrix_rank(res_pow) == 2);
}


int main(int argc, char *argv[]) {
assert_test();



matrix<double, 3, 5> mat1;
matrix<double, 5, 3> mat2;

const matrix<double, 3, 3> res_pow = \
{{{468, 576, 684}, {1062, 1305, 1548}, {1656, 2034, 2412}}};

fill_mat_(mat1, 2.);
fill_mat_(mat2, 4.);

std::cout << "sec diagonal -> 2\n";
show_mat(set_sec_diagonal(mat2, 2));

std::cout << "sec above -> 1\n";
show_mat(set_above_sec_diagonal(mat1, 1));

std::cout << "sec under -> 0\n";
show_mat(set_under_sec_diagonal(mat1, 0));

std::cout << "rank of square3x3: " << matrix_rank(res_pow) << "\n";

return 0;
}