
#include "maxmin.h"

int MaxMin::maxmin(vector<vector<int>> matrix, int threads, bool omp, bool log_time) {
int total_maxmin = matrix[0][0];
int dim = matrix.size();
omp_set_max_active_levels(1);
auto start = chrono::system_clock::now();
if (omp) {
#pragma omp parallel for shared(matrix, total_maxmin, dim) default(none) num_threads(threads)
{
for (int i = 0; i < dim; i++) {
int row_min = matrix[i][0];
for (int j = 0; j < dim; j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
}
else {
for (int i = 0; i < matrix.size(); i++) {
int row_min = matrix[i][0];
for (int j = 0; j < matrix[i].size(); j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
if (log_time) {
Utils->write_log_time(total_time, threads);
}
printf("Maxmin: %d\nExecution time: %ld ms\n", total_maxmin, total_time);
return total_maxmin;
}

int MaxMin::maxmin_dynamic(vector<vector<int>> matrix, bool log_time, int threads, int chunk) {
int total_maxmin = matrix[0][0];
auto start = chrono::system_clock::now();
#pragma omp parallel shared(matrix, total_maxmin, chunk) default(none) num_threads(threads)
{
#pragma omp for schedule(dynamic, chunk) nowait
for (int i = 0; i < matrix.size(); i++) {
int row_min = matrix[i][0];
for (int j = 0; j < matrix[i].size(); j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Maxmin: %d\n"
"Execution time: %ld ms\n", total_maxmin, total_time);
return total_maxmin;
}

int MaxMin::maxmin_guided(vector<vector<int>> matrix, bool log_time, int threads, int chunk) {
int total_maxmin = matrix[0][0];
auto start = chrono::system_clock::now();
#pragma omp parallel shared(matrix, total_maxmin, chunk) default(none) num_threads(threads)
{
#pragma omp for schedule(guided, chunk) nowait
for (int i = 0; i < matrix.size(); i++) {
int row_min = matrix[i][0];
for (int j = 0; j < matrix[i].size(); j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Maxmin: %d\n"
"Execution time: %ld ms\n", total_maxmin, total_time);
return total_maxmin;
}

int MaxMin::maxmin_static(vector<vector<int>> matrix, bool log_time, int threads, int chunk) {
int total_maxmin = matrix[0][0];
auto start = chrono::system_clock::now();
#pragma omp parallel shared(matrix, total_maxmin, chunk) default(none) num_threads(threads)
{
#pragma omp for schedule(guided, chunk) nowait
for (int i = 0; i < matrix.size(); i++) {
int row_min = matrix[i][0];
for (int j = 0; j < matrix[i].size(); j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Maxmin: %d\n"
"Execution time: %ld ms\n", total_maxmin, total_time);
return total_maxmin;
}

int MaxMin::calculate_nested(vector<vector<int>> matrix, int threads) {
int total_maxmin = matrix[0][0];
int dim = matrix.size();
omp_set_max_active_levels(2);
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(matrix, total_maxmin, dim) default(none) num_threads(threads) collapse(1)
{
for (int i = 0; i < dim; i++) {
int row_min = matrix[i][0];
for (int j = 1; j < dim; j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Maxmin: %d\nExecution time: %ld ms\n", total_maxmin, total_time);
return total_maxmin;
}

int MaxMin::calculate_nested2(vector<vector<int>> matrix, int threads) {
int total_maxmin = matrix[0][0];
int dim = matrix.size();
omp_set_max_active_levels(3);
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(matrix, total_maxmin, dim) default(none) num_threads(threads)
{
for (int i = 0; i < dim; i++) {
int row_min = matrix[i][0];
int parts_num = 90;
for (int k = 0; k < parts_num; k++) {
int part = (int) dim / parts_num;
for (int j = part * k; j < dim - part * (parts_num - k - 1); j++) {
if (matrix[i][j] < row_min) {
row_min = matrix[i][j];
}
}
if (row_min > total_maxmin) {
total_maxmin = row_min;
}
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Maxmin: %d\nExecution time: %ld ms\n", total_maxmin, total_time);
return total_maxmin;
}