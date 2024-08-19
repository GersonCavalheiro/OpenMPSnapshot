#include <iostream>
#include <string>

#include <fstream>
#include <sstream>
#include <vector>

#include "stdc++.h"
#include <array>
#include "/usr/local/opt/libomp/include/omp.h"

using namespace std;

#define NUM_THREADS 16
#define PARTITION_LEFT 5
#define PARTITION_RIGHT 6

enum direction
{
d_down,
d_right,
none
};

#define COORD std::pair<int, int>


int iter = 0;


void display_arr(int *arr, int n)
{

cout << "arr: ";

for (int i = 0; i < n; i++)
{
cout << arr[i] << " ";
}

cout << endl;
}

void print_coords(COORD start, COORD end)
{

cout << "Start:" << start.first << "," << start.second << endl;
cout << "End:" << end.first << "," << end.second << endl;
}

int find_length(COORD start, COORD end, direction dir)
{

if (dir == d_down)
return end.first - start.first;
if (dir == d_right)
return end.second - start.second;

return -1;
}

void convert_sol(int **mat, int **&sol_mat, int m, int n)
{

sol_mat = new int *[m]; 
for (int i = 0; i < m; i++)
{
sol_mat[i] = new int[n]; 
}

for (int i = 0; i < m; i++)
{
for (int j = 0; j < m; j++)
{
if (mat[i][j] == -2)
sol_mat[i][j] = -2; 
else
sol_mat[i][j] = -1; 
}
}
}

void print_one_matrix(int **matrix, int m, int n)
{
std::cout << "Matrix: " << std::endl;
for (int i = 0; i < m; i++)
{ 
for (int j = 0; j < n; j++)
{ 
std::cout << matrix[i][j] << "\t";
}
std::cout << "\n";
}
}

void sol_to_file(int **mat, int **sol_mat, int m, int n, string fname)
{

ofstream to_write(fname);

to_write << m << " " << n << "\n";

for (int i = 0; i < m; i++)
{
for (int j = 0; j < n; j++)
{
if (mat[i][j] != -2)
to_write << mat[i][j] << " ";
else
to_write << sol_mat[i][j] << " ";
}
to_write << "\n";
}

to_write.close();
}

void read_matrix(int **&matrix, std::ifstream &afile, int m, int n)
{

matrix = new int *[m]; 

for (int i = 0; i < m; i++)
{
matrix[i] = new int[n]; 
}

int val;
for (int i = 0; i < m; i++)
{
for (int j = 0; j < n; j++)
{
afile >> val;
matrix[i][j] = val;
}
}
}


struct sum
{
COORD start;
COORD end;

int hint;
int dir;
int length;
int *arr;

void print_sum()
{
cout << "############################" << endl;
cout << "Creating sum with: " << endl;
print_coords(start, end);
cout << "Hint: " << hint << endl;
cout << "Direction: " << dir << endl;
cout << "Length: " << length << endl;
cout << "############################" << endl;
}

sum(COORD _start, COORD _end, int _hint, direction _dir) : start(_start), end(_end), hint(_hint), dir(_dir)
{
length = find_length(_start, _end, _dir);
arr = new int[length];
#ifdef DEBUG
cout << "############################" << endl;
cout << "Creating sum with: " << endl;
print_coords(start, end);
cout << "Hint: " << hint << endl;
cout << "Direction: " << dir << endl;
cout << "Length: " << length << endl;
cout << "############################" << endl;
#endif
}

};

COORD find_end(int **matrix, int m, int n, int i, int j, direction dir)
{ 

if (dir == d_right)
{
for (int jj = j + 1; jj < n; jj++)
{
if (matrix[i][jj] != -2 || jj == n - 1)
{
if (matrix[i][jj] == -2 && jj == n - 1)
jj++;
COORD END = COORD(i, jj);
return END;
}
}
}

if (dir == d_down)
{
for (int ii = i + 1; ii < m; ii++)
{
if (matrix[ii][j] != -2 || ii == m - 1)
{
if (matrix[ii][j] == -2 && ii == m - 1)
ii++;
COORD END = COORD(ii, j);
return END;
}
}
}
}

vector<sum> get_sums(int **matrix, int m, int n)
{

vector<sum> sums;

for (int i = 0; i < m; i++)
{
for (int j = 0; j < n; j++)
{
int val = matrix[i][j];
if (val != -1 && val != -2)
{
int hint = val;
hint = hint / 10;

if ((hint % 100) == 0)
{
hint = (int)(hint / 100);
COORD START = COORD(i, j + 1);
COORD END = find_end(matrix, m, n, i, j, d_right);
sum _sum = sum(START, END, hint, d_right);
sums.push_back(_sum);
}

else
{
int div = (int)(hint / 100);
int rem = (int)(hint % 100);

if (div == 0 && rem != 0)
{
COORD START = COORD(i + 1, j);
COORD END = find_end(matrix, m, n, i, j, d_down);
sum _sum = sum(START, END, rem, d_down);
sums.push_back(_sum);
}

if (div != 0 && rem != 0)
{
COORD START1 = COORD(i + 1, j);
COORD START2 = COORD(i, j + 1);
COORD END1 = find_end(matrix, m, n, i, j, d_down);
COORD END2 = find_end(matrix, m, n, i, j, d_right);
sum _sum1 = sum(START1, END1, rem, d_down);
sum _sum2 = sum(START2, END2, div, d_right);
sums.push_back(_sum1);
sums.push_back(_sum2);
}
}
}
}
}
return sums;
}

enum sumStatus
{
success,
over,
under,
duplicate
};

sumStatus checkSumStatus(int remaining_sum, int remaining_cells)
{
int current_max_num = 9;
int current_min_num = 1;
int max_num = 0;
int min_num = 0;

for (int i = 0; i < remaining_cells; i++)
{
max_num += current_max_num;
min_num += current_min_num;
current_max_num--;
current_min_num++;
}

if (remaining_sum > max_num)
return sumStatus::under;

if (remaining_sum < min_num)
return sumStatus::over;

return sumStatus::success;
}


sumStatus checkSum(int **sol_mat, sum _sum)
{
int hint = _sum.hint;
int row_idx = _sum.start.first;
int col_idx = _sum.start.second;

vector<bool> checks(9, false);

if (_sum.dir == direction::d_right)
{
int end_idx = _sum.end.second;

while (col_idx < end_idx && sol_mat[row_idx][col_idx] > 0)
{
hint -= sol_mat[row_idx][col_idx];
sumStatus status = checkSumStatus(hint, end_idx - col_idx - 1);
if (status != sumStatus::success)
return status;

if (checks[sol_mat[row_idx][col_idx]])
return sumStatus::duplicate;

checks[sol_mat[row_idx][col_idx]] = true;
col_idx++;
}
}

else
{
int end_idx = _sum.end.first;

while (row_idx < end_idx && sol_mat[row_idx][col_idx] > 0)
{
hint -= sol_mat[row_idx][col_idx];
sumStatus status = checkSumStatus(hint, end_idx - row_idx - 1);
if (status != sumStatus::success)
return status;

if (checks[sol_mat[row_idx][col_idx]])
return sumStatus::duplicate;

checks[sol_mat[row_idx][col_idx]] = true;
row_idx++;
}
}

return sumStatus::success;
}

vector<vector<vector<sum *>>> setCell2Sums(vector<sum> &sums, int m, int n)
{
vector<vector<vector<sum *>>> cell_2_sums(m, vector<vector<sum *>>(n, vector<sum *>()));

for (int i = 0; i < sums.size(); i++)
{
int start_row = sums[i].start.first;
int start_col = sums[i].start.second;
int end_row = sums[i].end.first;
int end_col = sums[i].end.second;

sum *tmp = &(sums[i]);
if (sums[i].dir == direction::d_right)
{
for (int j = start_col; j < end_col; j++)
{
cell_2_sums[start_row][j].push_back(tmp);
}
}
else
{
for (int j = start_row; j < end_row; j++)
{
cell_2_sums[j][start_col].push_back(tmp);
}
}
}

return cell_2_sums;
}

int **copyMatrix(int **&mat, int m, int n)
{
int **copy = new int *[m];
for (int i = 0; i < m; i++)
{
copy[i] = new int[n];
for (int j = 0; j < n; j++)
{
copy[i][j] = mat[i][j];
}
}

return copy;
}

bool checkEnd(const int &direction, const int &v)
{
if (direction == -1)
return v > 0;
return v < 10;
}

int **kakuro_task(int **sol_mat, int k, int m, int n, vector<vector<vector<sum *>>> &cell_2_sums, int direction, int start_from)
{
int i = std::ceil(k / m);
int j = k % n;

while (sol_mat[i][j] != -2 && k < m * n)
{
if (k == m * n - 1)
{
return copyMatrix(sol_mat, m, n);
}
k++;
i = std::ceil(k / m);
j = k % n;
}

for (int v = start_from; checkEnd(direction, v); v += direction)
{
sol_mat[i][j] = v;
vector<sum *> sums = cell_2_sums[i][j];
bool is_valid = true;

if (sums.size() > 0)
{
sumStatus status = checkSum(sol_mat, *sums[0]);

if (direction == 1 && status == sumStatus::over)
return nullptr;

if (direction == 0 && status == sumStatus::under)
return nullptr;

if (status != sumStatus::success)
is_valid = false;
}

if (sums.size() > 1)
{
sumStatus status = checkSum(sol_mat, *sums[1]);

if (direction == 1 && status == sumStatus::over)
return nullptr;

if (direction == 0 && status == sumStatus::under)
return nullptr;

if (status != sumStatus::success)
is_valid = false;
}

if (is_valid)
{
if (k == m * n - 1)
return copyMatrix(sol_mat, m, n);

int **sol_copy_l = copyMatrix(sol_mat, m, n);
int **l = kakuro_task(sol_copy_l, k + 1, m, n, cell_2_sums, -1, PARTITION_LEFT);

for (int i = 0; i < m; i++)
{
delete[] sol_copy_l[i];
}
delete[] sol_copy_l;

if (l)
return l;

int **sol_copy_r = copyMatrix(sol_mat, m, n);
int **r = kakuro_task(sol_copy_r, k + 1, m, n, cell_2_sums, 1, PARTITION_RIGHT);

for (int i = 0; i < m; i++)
{
delete[] sol_copy_r[i];
}
delete[] sol_copy_r;

if (r)
return r;
}
}
return nullptr;
}

bool solution(int **mat, int **&sol_mat, vector<sum> sums, int m, int n)
{

vector<vector<vector<sum *>>> cell_2_sums = setCell2Sums(sums, m, n);

int **copy1 = copyMatrix(sol_mat, m, n);
int **copy2 = copyMatrix(sol_mat, m, n);

int **l = kakuro_task(copy1, 0, m, n, cell_2_sums, -1, PARTITION_LEFT);

int **r = kakuro_task(copy2, 0, m, n, cell_2_sums, 1, PARTITION_RIGHT);

#ifdef LEAK_DEBUG
int empty;
cout << "Checking for leaks...";
cin >> empty;
#endif

if (l)
{
sol_mat = l;
return true;
}
if (r)
{
sol_mat = r;
return true;
}


return false;
}

int main(int argc, char **argv)
{

std::string filename(argv[1]);
std::ifstream file;
file.open(filename.c_str());

int m, n;
file >> m;
file >> n;

int **mat;
read_matrix(mat, file, m, n);
print_one_matrix(mat, m, n);

int **sol_mat;
convert_sol(mat, sol_mat, m, n);
print_one_matrix(sol_mat, m, n);

vector<sum> sums = get_sums(mat, m, n);

double start = omp_get_wtime();
solution(mat, sol_mat, sums, m, n);
double end = omp_get_wtime();

print_one_matrix(sol_mat, m, n);
sol_to_file(mat, sol_mat, m, n, "solution.kakuro");

cout << "Time taken: " << ((end - start) * 1000) << " (ms)." << endl;
for (int i = 0; i < n; i++)
{
delete mat[i];
delete sol_mat[i];
}

delete mat;
delete sol_mat;

return 0;
}
