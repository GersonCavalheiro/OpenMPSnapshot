




#include <iostream>
#include <amp.h>
#include <omp.h>
#include <ctime>
#include <cstdlib>
#include "timer.h"
#include <array>
#include <vector>


using namespace concurrency;
using namespace std;

const size_t N = 5000;

int aMatrix2D[N][N];
int bMatrix2D[N][N];
int resMatrix2D[N][N];

int aMatrix[N*N];
int bMatrix[N*N];
int resMatrix[N*N];

std::vector<int> aVector(N);
std::vector<int> bVector(N);
std::vector<int> resVector(N);


LARGE_INTEGER Timer::m_freq = \
(QueryPerformanceFrequency(&Timer::m_freq), Timer::m_freq);


LONGLONG Timer::m_overhead = Timer::GetOverhead();


void information_of_system()

{
auto all = accelerator::get_all();
for (size_t i = 0; i < all.size(); i++)
{
auto acc = all[i];
wcout << acc.description << endl;
wcout << acc.device_path << endl;
wcout << acc.get_is_emulated() << endl;
wcout << acc.dedicated_memory << endl;
wcout << acc.get_is_debug() << endl;
}
}


void Addition(vector<int>& a, vector<int>& b, vector<int>& vec) {
for (std::size_t i = 0U; i < N; i++)
vec[i] = a[i] + b[i];
}


void AdditionWithOpenMP(vector<int>& a, vector<int>& b, vector<int>& vec) {

int threadsNum = 2;
omp_set_num_threads(threadsNum);
int i;

#pragma omp parallel for shared(a, b, vec) private(i)
for (i = 0; i < N; i++)
vec[i] = a[i] + b[i];
}


void AdditionWithAMP(vector<int>& aV, vector<int>& bV, vector<int>& vec) {
array_view<const int, 1> a(N, aV);
array_view<const int, 1> b(N, bV);
array_view<int, 1> product(N, vec);
product.discard_data();

parallel_for_each(product.extent,
[=](index<1> idx) restrict(amp) {
product[idx] = a[idx] + b[idx];
}
);
}

void TransposeWithout(int aM[][N], int resM[][N]) {

int i, j;

for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
resM[j][i] = aM[i][j];
}
}
}

void TransposeWithOpenMP(int aM[][N], int resM[][N]) {

int threadsNum = 2;
omp_set_num_threads(threadsNum);
int i, j;

#pragma omp parallel for shared(aM) private(i, j)
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
resM[j][i] = aM[i][j];
}
}
}

void TransposeWithAMP(int aM[], int resM[]) {
array_view<const int, 2> a(N, N, aM);
array_view<int, 2> res(N, N, resM);
res.discard_data();

parallel_for_each(res.extent,
[=](index<2> idx) restrict(amp) {
res(idx[0], idx[1]) = a(idx[1], idx[0]);
});
}


void MultiplyByNumberSimply(int aM[], int resM[]) {
int i, j;
const int number = 500;

for (i = 0; i < N; i++) 
for (j = 0; j < N; j++) 
resM[i + j] = (aM[i + j] * number);
}

void MultiplyByNumberWithOpenMP(const int aM[], int resM[]) {

int threadsNum = 4;
omp_set_num_threads(threadsNum);
int i, j;
const int number = 500;

#pragma omp parallel for shared(aM) private(i, j)
for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) 
resM[i + j] = (aM[i + j] * number);
}
}

void MultiplyByNumberWithAMP(int aMatrix[], int resMatrix[]) {
array_view<const int, 2> a(N, N, aMatrix);
array_view<int, 2> res(N, N, resMatrix);
res.discard_data();

const int number = 500;

parallel_for_each(res.extent,
[=](index<2> idx) restrict(amp) {
res[idx] = a[idx] * number;
});

res.synchronize();
}

void MultiplySimply(int aMatrix[], int bMatrix[], int resMatrix[]) {
int i, j, k;

for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
for (k = 0; k < N; k++) {
resMatrix[i+j] += (aMatrix[i+k] * bMatrix[k+j]);
}
}
}
}

void MultiplyWithOpenMP(const int aMatrix[], const int bMatrix[], int resMatrix[]) {


int threadsNum = 2;
omp_set_num_threads(threadsNum);
int i, j, k;

#pragma omp parallel for shared(aMatrix, bMatrix, resMatrix) private(i, j, k)

for (i = 0; i < N; i++) {
for (j = 0; j < N; j++) {
for (k = 0; k < N; k++) {
resMatrix[i + j] += (aMatrix[i + k] * bMatrix[k + j]);
}
}
}
}

void MultiplyWithAMP(int aMatrix[], int bMatrix[], int resMatrix[]) {

array_view<int, 2> a(N, N, aMatrix);
array_view<int, 2> b(N, N, bMatrix);
array_view<int, 2> product(N, N, resMatrix);
product.discard_data();

parallel_for_each(product.extent,
[=](index<2> idx) restrict(amp) {
int row = idx[0];
int col = idx[1];
for (int inner = 0; inner < 2; inner++) {
product[idx] += a(row, inner) * b(inner, col);
}
});
product.synchronize();
}

void FillMatrix(int arr[]) {
for (int i = 0; i < N*N; i++)
arr[i]= rand() % 3;
}

void FillMatrixZeros(int arr[]) {
for (int i = 0; i < N*N; i++)
arr[i] = 0;
}

void FillMatrix2D(int arr[][N]) {
for (int i = 0; i < N; i++)
for (int j = 0; i < N; i++)
arr[i][j] = rand() % 3;
}

void FillMatrixZeros2D(int arr[][N]) {
for (int i = 0; i < N; i++)
for (int j = 0; i < N; i++)
arr[i][j] = 0;
}

void Show(int arr[]) {
for (int i = 0; i < 10; i++) {
for (int j = 0; j < 10; j++)
std::cout << arr[i + j] << " ";
std::cout << "\n" << " ";
}
}

void Show2D(int arr[][N]) {
for (int i = 0; i < N; i++) {
for (int j = 0; j < N; j++)
std::cout << arr[i][j] << " ";
std::cout << "\n" << " ";
}
}

void show_vector(vector<int>& vec) {
std::cout << "\nВывод вектора\n" << " ";
for (std::size_t i = 0U; i < 10; i++)
std::cout << vec[i]<< " ";
}

void fill_vector(vector<int>& vec) {
for(size_t i = 0; i < N; i++)
vec[i] = rand() % 2;
}

void fill_zeros_vector(vector<int>& vec) {
for (size_t i = 0; i < N; i++)
vec[i] = 0;
}

void AdditionVector() {
Timer t;
double time_value;

fill_vector(aVector);
fill_vector(bVector);
fill_zeros_vector(resVector);




t.Start();
Addition(aVector, bVector, resVector);
t.Stop();

time_value = t.Elapsed();
std::cout << "Сложение векторов:" << "  ";
std::cout << time_value << "\n";


fill_zeros_vector(resVector);
t.Start();
AdditionWithOpenMP(aVector, bVector, resVector);
t.Stop();

time_value = t.Elapsed();
std::cout << "Сложение векторов c OpenMP:" << "  ";
std::cout << time_value << "\n";


fill_zeros_vector(resVector);
t.Start();
AdditionWithAMP(aVector, bVector, resVector);
t.Stop();

time_value = t.Elapsed();
std::cout << "Сложение векторов c AMP:" << "  ";
std::cout << time_value << "\n";

}

void MatrixTranspose() {
Timer t;
double time_value;

FillMatrix2D(aMatrix2D);
FillMatrixZeros2D(bMatrix2D);

t.Start();
TransposeWithout(aMatrix2D, bMatrix2D);
t.Stop();

time_value = t.Elapsed();
std::cout << "Транспонирование матрицы" << "  ";
std::cout << time_value << "\n";

FillMatrixZeros2D(bMatrix2D);

t.Start();
TransposeWithOpenMP(aMatrix2D, bMatrix2D);
t.Stop();

time_value = t.Elapsed();
std::cout << "Транспонирование матрицы с OpenMP" << "  ";
std::cout << time_value << "\n";

FillMatrix(aMatrix);
FillMatrixZeros(bMatrix);

t.Start();
TransposeWithAMP(aMatrix, bMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Транспонирование матрицы с AMP" << "  ";
std::cout << time_value << "\n";


}

void MatrixMultiplicationByNumber() {
Timer t;
double time_value;

FillMatrix(aMatrix);
FillMatrixZeros(bMatrix);

t.Start();
MultiplyByNumberSimply(aMatrix, bMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Умножение матрицы  на число" << "  ";
std::cout << time_value << "\n";

FillMatrixZeros(bMatrix);

t.Start();
MultiplyByNumberWithOpenMP(aMatrix, bMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Умножение матрицы  на число с OpenMP" << "  ";
std::cout << time_value << "\n";

FillMatrixZeros(bMatrix);

t.Start();
MultiplyByNumberWithAMP(aMatrix, bMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Умножение матрицы  на число с AMP" << "  ";
std::cout << time_value << "\n";

}

void MatrixMultiplication() {
Timer t;
double time_value;

FillMatrix(aMatrix);
FillMatrix(bMatrix);
FillMatrixZeros(resMatrix);

t.Start();
MultiplySimply(aMatrix, bMatrix, resMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Умножение матриц" << "  ";
std::cout << time_value << "\n";

FillMatrixZeros(resMatrix);
t.Start();
MultiplyWithOpenMP(aMatrix, bMatrix, resMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Умножение матриц с OpenMP" << "  ";
std::cout << time_value << "\n";

FillMatrix(aMatrix);
FillMatrix(bMatrix);
FillMatrixZeros(resMatrix);

t.Start();
MultiplyWithAMP(aMatrix, bMatrix, resMatrix);
t.Stop();

time_value = t.Elapsed();
std::cout << "Умножение матриц с AMP" << "  ";
std::cout << time_value << " ";
}

int main(){   

setlocale(LC_ALL, "Russian");

information_of_system();
}

