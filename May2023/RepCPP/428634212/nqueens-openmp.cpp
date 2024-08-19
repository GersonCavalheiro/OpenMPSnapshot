#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <stack>
#include <thread>
#include <omp.h>
#include <algorithm>

#define N_MAX 12


inline bool boardIsValidSoFar(int lastPlacedRow, const int* gameBoard, const int N)
{
int lastPlacedColumn = gameBoard[lastPlacedRow];



for (int row = 0; row < lastPlacedRow; ++row)
{


if (gameBoard[row] == lastPlacedColumn) 

return false;
const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
if (gameBoard[row] == col1 || gameBoard[row] == col2)
return false;
}
return true;
}

void calculateSolutions(int N, std::vector<std::vector<int>>& solutions)
{
const long long int O = powl(N, N);


int* solutions_array = (int*)malloc(pow(N, 5) * sizeof(int)); 
std::atomic<int> num_solutions = 0;

#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(static) 
for (long long int i = 0; i < O; i++) {
bool valid = true;
int gameBoard[N_MAX]; 

long long int column = i;
for (int j = 0; j < N; j++) {
gameBoard[j] = column % N;

if (!boardIsValidSoFar(j, gameBoard, N)) {
valid = false;
break;
}

column /= N;
}

if (valid) {



for (int j = 0; j < N; j++)
solutions_array[N * num_solutions + j] = gameBoard[j];
num_solutions++;
}
}


for (int i = 0; i < num_solutions; i++) {
std::vector<int> solution = std::vector<int>();
for (int j = 0; j < N; j++)
solution.push_back(solutions_array[N * i + j]);
solutions.push_back(solution);
}
free(solutions_array);
}

void calculateAllSolutions(int N, bool print)
{
std::vector<std::vector<int>> solutions;

auto start = omp_get_wtime();
calculateSolutions(N, solutions);
auto stop = omp_get_wtime();

auto time_elapsed = stop - start;
std::cout << "N=" << N << " time elapsed: " << time_elapsed << "s\n";
printf("N=%d, solutions=%d\n\n", N, int(solutions.size()));

if (print)
{
std::string text;
text.resize(N * (N + 1) + 1); 
text.back() = '\n'; 
for (const auto& solution : solutions)
{
for (int i = 0; i < N; ++i)
{
auto queenAtRow = solution[i];
for (int j = 0; j < N; ++j)
text[i * (N + 1) + j] = queenAtRow == j ? 'X' : '.';
text[i * (N + 1) + N] = '\n';
}
std::cout << text << "\n";
}
}
}


int main(int argc, char** argv)
{
for (int N = 4; N <= N_MAX; ++N)
calculateAllSolutions(N, false);
}