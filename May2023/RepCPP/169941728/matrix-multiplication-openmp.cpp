
#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include <omp.h>
#include <random>

using namespace std;

#define MATRIX_SIZE 1000

vector<int> result(MATRIX_SIZE, 0);


void populateVectorRandom(vector<int> &A, double m, int flag) {
random_device rd;
default_random_engine e(rd());
uniform_int_distribution<> range1(0, m);
uniform_int_distribution<> range2(0, (int) A.size() / 50);
uniform_int_distribution<> range3(0, 10);

if (flag == 1) {
for (double i = 0; i < A.size(); i++) {
A[i] = range1(e);
}
}
else if (flag == 2) {
for (double i = 0; i < A.size(); i++) {
A[i] = i + range3(e) + 1;
}
}
else if (flag == 3) {
for (double i = 0; i < A.size(); i++) {
A[i] = A.size() * 2 - range3(e) - i;
}
}
else if (flag == 4) {
for (double i = 0; i < A.size(); i++) {
A[i] = range3(e);
}
}
}


void printVector(vector<int> A) {
int n = 0;
for (int &i : A) {
if (n >= 1001) {
cout << "only showing " << n << " elements, omit remaining......";
break;
}
cout << i << "\t";

if (++n % 10 == 0)
cout << "\n";
}
cout << endl;
}

void printMatrix(vector<vector<int>> A) {
int n = 0;
for (vector<int> j : A) {
for (int &i : j) {
if (n >= 1001) {
cout << "only showing " << n << " elements, omit remaining......";
break;
}
cout << i << "\t";

if (++n % 10 == 0)
cout << "\n";
}
cout << endl;
}
}

int main() {

cout << "\n********** CPU Information **********" << endl;
cout << "Number of CPU cores: " << omp_get_num_procs() << endl;

cout << "\n********** Matrix-Vector Multiplication **********" << endl;
vector<vector<int>> X;
vector<int> Y(MATRIX_SIZE, 0);

cout << "Initializing Matrix X...";
for (int i = 0; i < MATRIX_SIZE; i++) {
vector<int> row(MATRIX_SIZE, 0);
populateVectorRandom(row, 1, 1);
X.push_back(row);
}
cout << "DONE" << endl;

cout << "Initializing Matrix Y...";
populateVectorRandom(Y, 1, 1);
cout << "DONE" << endl;

cout << "Running the Multiplication between X and Y..." << endl;
int i, j;
int sum = 0;
#pragma omp parallel shared(X, Y, result) private(i, j)
{
cout << "Thread " << omp_get_thread_num() << " is running" << endl;
for (i = 0; i < MATRIX_SIZE; i++) {
#pragma omp parallel for reduction (+:sum)
for (j = 0; j < MATRIX_SIZE; j++)
sum += X[i][j] * Y[j];
result[i] = sum;
sum = 0;
}
}
cout << "\nDONE" << endl;

cout << "\n********** Resulting Vector **********\n";
printVector(result);
cout << "********** Exit **********\n";

return 0;
}
