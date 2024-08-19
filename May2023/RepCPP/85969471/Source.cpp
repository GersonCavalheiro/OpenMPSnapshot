#include <fstream>
#include <iostream>
#include <ctime>
#include <string>

#define SIZE 1000 

using namespace std;

void createMatrix(string nameTxt) {
srand(time(0));

ofstream fout(nameTxt, ios_base::out | ios_base::trunc);

for (int i = 0; i < SIZE; i++)
{
for (int j = 0; j < SIZE; j++)
{
fout << rand() << " ";

if (j < SIZE-1)
{
fout << " ";
}
else if (i < SIZE)
{
fout << endl;
}
}
}
}

void writeTime(double t[SIZE]) {
srand(time(0));

ofstream fout("result_time1.txt", ios_base::out | ios_base::trunc);

for (int i = 0; i < SIZE; i++)
{
fout << t[i] << endl;
}
fout.close();
}

void askToCreateMatrix() {
char answer;
cin >> answer;
if (answer == 'y')
{
createMatrix("matrix_1.txt");
createMatrix("matrix_2.txt");
}
else if (answer == 'n') {}
else
{
cout << "  .   (y/n): ";
askToCreateMatrix();
}
}

void readMatrixFromTxt(string nameTxt, int **matrix)
{
ifstream fin(nameTxt, ios_base::in);
int number;
for (int i = 0; i < SIZE; i++) {
for (int j = 0; j < SIZE; j++) {
fin >> (int) matrix[i][j];
}
}
fin.close();
}

void printMatrix(int **matrix, int N) {
for (size_t i = 0; i < N; i++)
{
for (size_t j = 0; j < N; j++)
{
cout << "\t" << matrix[i][j];
}
cout << endl;
}
}

void initializeMatrix(int **matrix, int N) {
for (size_t i = 0; i < N; i++)
{
for (size_t j = 0; j < N; j++)
{
matrix[i][j]=0;
}
}
}

int** matrixProduct(int **matrix_1, int**matrix_2, int isParallel) {
int **matrix_result = new int *[SIZE];
for (int i = 0; i < SIZE; i++) {
matrix_1[i] = new int[SIZE];
matrix_2[i] = new int[SIZE];
matrix_result[i] = new int[SIZE];
}
initializeMatrix(matrix_result, SIZE);
readMatrixFromTxt("matrix_1.txt", matrix_1);
readMatrixFromTxt("matrix_2.txt", matrix_2);

double timeArray[SIZE];
clock_t t1, t2;

for (size_t size = 2; size < SIZE; size++)
{
int i, j, k;
t1 = clock();
#pragma omp parallel for private(j,k) if(isParallel)	
for (i = 0; i<size; i++)
{
for (j = 0; j<size; j++)
for (k = 0; k<size; k++)
matrix_result[i][j] += matrix_1[i][k] * matrix_2[k][j];
}
t2 = clock();
timeArray[size] = (double)(t2 - t1) / (double)CLOCKS_PER_SEC;
if ((size == 100) || (size == 200) || (size == 300) ||
(size == 400) || (size == 500) || (size == 600) ||
(size == 700) || (size == 800) || (size == 900) || (size == 999))
{
cout << "size of matrix: " << size << " Time: " << timeArray[size] << '\n';
}
}
writeTime(timeArray);
ofstream fout("resultMatrix1.txt", ios_base::out | ios_base::trunc);
for (int i = 0; i < SIZE; i++)
{
for (int j = 0; j < SIZE; j++)
{
fout << matrix_result[i][j] << " ";
if (j < SIZE - 1)
{
fout << " ";
}
else if (i < SIZE)
{
fout << endl;
}
}
}
return matrix_result;
}

int main() {
setlocale(LC_ALL, "ru");
cout << "  ? (y/n): ";
askToCreateMatrix();

int **matrix_1 = new int*[SIZE];
int **matrix_2 = new int*[SIZE];

for (int i = 0; i < SIZE; i++) {
matrix_1[i] = new int[SIZE];
matrix_2[i] = new int[SIZE];
}

readMatrixFromTxt("matrix_1.txt", matrix_1);
readMatrixFromTxt("matrix_2.txt", matrix_2);

matrixProduct(matrix_1, matrix_2, 1);
matrixProduct(matrix_1, matrix_2, 0);

system("pause");
}