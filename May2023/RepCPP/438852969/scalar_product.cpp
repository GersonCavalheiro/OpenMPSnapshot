
#include "scalar_product.h"


int ScalarProduct::calculate_simple(vector <int> array1, vector <int> array2) {
int scalar_product = 0;

for (int i = 0; i < min(array1.size(), array2.size()); i++) {
scalar_product = scalar_product + array1[i] * array2[i];
}
printf("Scalar product of arrays: %d\n", scalar_product);
return scalar_product;
}

int ScalarProduct::calculate(const int array1[], const int array2[], int array_length, int threads) {
int scalar_product = 0;
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(array1, array2, array_length) reduction(+:scalar_product) default(none) num_threads(threads)
{
for (int i = 0; i < array_length; i++) {
scalar_product = scalar_product + array1[i] * array2[i];
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Scalar product of arrays: %d\n"
"Execution time: %ld ms\n", scalar_product, total_time);
return scalar_product;
}

int ScalarProduct::calculate_atomic(vector<int> array1, vector<int> array2, int threads) {
int scalar_product = 0;
int prod;
int i;
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(array1, array2, scalar_product) private(prod, i) num_threads(threads)
{
for (i = 0; i < array1.size(); i++) {
prod = array1[i] * array2[i];
#pragma omp atomic
scalar_product = scalar_product + prod;
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Scalar product of arrays: %d\n"
"Execution time: %ld ms\n", scalar_product, total_time);
return scalar_product;
}

int ScalarProduct::calculate_critical(vector<int> array1, vector<int> array2, int threads) {
int scalar_product = 0;
int prod;
int i;
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(array1, array2, scalar_product) private(prod, i) num_threads(threads)
{
for (i = 0; i < array1.size(); i++) {
prod = array1[i] * array2[i];
#pragma omp critical
scalar_product = scalar_product + prod;
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Scalar product of arrays: %d\n"
"Execution time: %ld ms\n", scalar_product, total_time);
return scalar_product;
}

int ScalarProduct::calculate_lock(vector<int> array1, vector<int> array2, int threads) {
omp_lock_t lock;
omp_init_lock(&lock);
int scalar_product = 0;
int prod;
int i;
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(array1, array2, scalar_product) private(prod, i) num_threads(threads)
{
for (i = 0; i < array1.size(); i++) {
prod = array1[i] * array2[i];
omp_set_lock (&lock);
scalar_product = scalar_product + prod;
omp_unset_lock (&lock);
}
}
omp_destroy_lock (&lock);
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Scalar product of arrays: %d\n"
"Execution time: %ld ms\n", scalar_product, total_time);
return scalar_product;
}

vector <int> ScalarProduct::calculate_sections(int length, int v_num, int threads) {
int i;
int j;
int scalar_product;
vector <int> total = {};
auto vectors = get_vectors(length, v_num);
for (i = 0; i < v_num; i++) {
total.push_back(0);
}
vector <int> v1 = {};
vector <int> v2 = {};
auto start = chrono::system_clock::now();
#pragma omp parallel shared(length, v_num, total, vectors) private(i, j, v1, v2, scalar_product) default(none)  num_threads(threads)
{
for (i = 0; i < v_num; i++) {
#pragma omp sections
{
#pragma omp section
{
v1 = vectors[i][0];
v2 = vectors[i][1];
}
#pragma omp section
{
scalar_product = 0;
for (j = 0; j < length; j++)
{
scalar_product += v1[j] * v2[j];
}
#pragma omp critical
{
total[i] = scalar_product;
}
}
}
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
print_data(vectors, total);
printf("Execution time: %ld ms\n", total_time);
return total;
}


void ScalarProduct::print_vector(vector <int> v1) {
cout << "  [";
for (int i = 0; i < v1.size(); i++) {
cout << v1[i] << ", ";
}
cout << "]\n";
}

void ScalarProduct::print_data(vector <vector <vector <int>>> vectors, vector <int> total) {
for (int i = 0; i < total.size(); i++) {
cout << i + 1 << ": \n";
print_vector(vectors[i][0]);
print_vector(vectors[i][1]);
cout << "  result: " << total[i] << "\n\n";
}
}

vector <vector <vector <int>>> ScalarProduct::get_vectors(int length, int v_num) {
vector <vector <vector <int>>> vectors = {};
for (int i = 0; i < v_num; i++) {
vectors.push_back({{}, {}});
for (int j = 0; j < length; j ++) {
vectors[i][0].push_back(rand() % 100 - 50);
vectors[i][1].push_back(rand() % 100 - 50);
}
}
return vectors;
}

int ScalarProduct::calculate2(vector<int> array1, vector<int> array2, int threads) {
int scalar_product = 0;
auto start = chrono::system_clock::now();
#pragma omp parallel for shared(array1, array2) reduction(+:scalar_product) default(none) num_threads(threads)
{
for (int i = 0; i < array1.size(); i++) {
scalar_product = scalar_product + array1[i] * array2[i];
}
}
auto end = chrono::system_clock::now();
long int total_time = chrono::duration_cast<chrono::milliseconds>(end - start).count();
printf("Scalar product of arrays: %d\n"
"Execution time: %ld ms\n", scalar_product, total_time);
return scalar_product;
}
