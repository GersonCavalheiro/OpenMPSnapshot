#include <chrono>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <vector>

using namespace std;

const int MAX_VALUE = 10;

vector<vector<int>> get_matrix(int height, int width)
{
vector<vector<int>> matrix = vector<vector<int>>(height, vector<int>(width, 0));

for (int i = 0; i < height; i++)
{
for (int j = 0; j < width; j++) {
matrix[i][j] = rand() % MAX_VALUE;
}
}
return matrix;
}

vector<int> get_vector(int width)
{
vector<int> vect = vector<int>(width, 0);

for (int i = 0; i < width; i++) {
vect[i] = rand() % MAX_VALUE;
}
return vect;
}

vector<int> calc_non_parallel(vector<vector<int>> matrix, vector<int> vect)
{
int width = matrix[0].size();

vector<int> result = vector<int>(width, 0);

for (int i = 0; i < width; i++) {
for (int j = 0; j < vect.size(); j++) {
result[i] += matrix[j][i] * vect[j];
}
}
return result;
}

vector<int> calc_parallel(vector<vector<int>> matrix, vector<int> vect)
{
int width = matrix[0].size();

vector<int> result = vector<int>(width, 0);

#pragma omp parallel for
for (int i = 0; i < width; i++) {
for (int j = 0; j < vect.size(); j++) {
result[i] += matrix[j][i] * vect[j];
}
}
return result;
}

void print_result(vector<int> vect, long time)
{
for (int i: vect)
printf("%d ", i);
printf("\n");
printf("Time: %d\n", time);
}

void compare_results(long time_np, long time_p)
{
if(time_np > time_p) {
printf("%d > %d\n", time_np, time_p);
} 
else {
printf("%d < %d\n", time_np, time_p);
}
}

int main() {

const int HEIGHT = 1000;
const int WIDTH = 15;

chrono::steady_clock::time_point start;
chrono::steady_clock::time_point end;

vector<vector<int>> matrix = get_matrix(HEIGHT, WIDTH);
vector<int> vect = get_vector(HEIGHT);
vector<int> result_vect;

start = chrono::steady_clock::now();
result_vect = calc_non_parallel(matrix, vect);
end = chrono::steady_clock::now();

long time_np = chrono::duration_cast<chrono::microseconds>(end - start).count();

printf("Result for non-parallel calculations: ");
print_result(result_vect, time_np);

start = chrono::steady_clock::now();
result_vect = calc_parallel(matrix, vect);
end = chrono::steady_clock::now();

long time_p = chrono::duration_cast<chrono::microseconds>(end - start).count();

printf("Result for parallel calculations: ");
print_result(result_vect, time_p);

compare_results(time_np, time_p);
}
