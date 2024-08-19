#include <omp.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <vector>

typedef double my_type;
auto constexpr num_runs = 50;
auto constexpr n = 2048;

int initialize_array(my_type**& matrix)
{
matrix = (my_type * *)malloc(sizeof(my_type*) * n);
if (matrix == nullptr)
{
std::cout << "Could not allocate memory for matrix!";
return 1;
}

for (auto i = 0; i < n; ++i)
{
matrix[i] = (my_type*)malloc(sizeof(my_type) * n);
if (matrix[i] == nullptr)
{
std::cout << "Could not allocate memory for matrix[" << i << "] !";
return 1;
}
}

for (auto i = 0; i < n; ++i)
for (auto j = 0; j < n; ++j)
matrix[i][j] = rand() % 10000;

return 0;
}

int main()
{
srand(time(NULL));
my_type** matrix = nullptr;

#pragma region Initialization
{
std::cout << "Initialization - ";

const auto t0 = std::chrono::high_resolution_clock::now();
if (initialize_array(matrix))
return 1;

const auto t1 = std::chrono::high_resolution_clock::now();
const auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

std::cout << "Init Duration =  " << duration_ms << "(ms)";
std::cout << std::endl;
}
#pragma endregion

my_type sum = 0;
my_type avg = 0;

int num_threads = omp_get_max_threads();
omp_set_num_threads(num_threads);

std::cout << "---------------------------" << std::endl;

#pragma region Sequential
{
std::cout << "Sequential -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

for (auto i = 0; i < n; ++i)
for (auto j = 0; j < n; ++j)
sum += matrix[i][j];

const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

avg = sum / (n * n);

const auto duration_ms = (sum_duration / num_runs) / 1000.0;
std::cout << "		Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}
#pragma endregion

std::cout << "---------------------------" << std::endl;

#pragma region Simple Parallel

{
std::cout << "Simple Parallel -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

#pragma omp parallel 
{
const auto id = omp_get_thread_num();

if (id == 0)
num_threads = omp_get_num_threads();

double _temp_sum = 0;
for (int i = (id / num_threads) * n; i < ((id + 1) / num_threads) * n; ++i)
for (int j = 0; j < n; ++j)
_temp_sum += matrix[i][j];
#pragma omp atomic
sum += _temp_sum;
}

avg = sum / (n * n);
const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

const auto duration_ms = (sum_duration / num_runs) / 1000.0;
std::cout << "	Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}

#pragma endregion

std::cout << "---------------------------" << std::endl;

#pragma region Simple Parallel (For Exchange)

{
std::cout << "Simple Parallel (For Exchange) -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

#pragma omp parallel
{
const auto id = omp_get_thread_num();

double _temp_sum = 0;
for (int j = 0; j < n; ++j)
for (int i = (id / num_threads) * n; i < ((id + 1) / num_threads) * n; ++i)
_temp_sum += matrix[i][j];
#pragma omp atomic
sum += _temp_sum;
}

avg = sum / (n * n);
const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

const auto duration_ms = (sum_duration / num_runs) / 1000.0;
std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}

#pragma endregion 

std::cout << "---------------------------" << std::endl;

#pragma region Parallel For Reduction

{
std::cout << "Parallel For Reduction -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

#pragma omp parallel for reduction(+: sum)
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
sum += matrix[i][j];

avg = sum / (n * n);
const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
std::cout << "	Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}

#pragma endregion

std::cout << "---------------------------" << std::endl;

#pragma region Parallel Simple Array - False Sharing
{
std::cout << "Parallel For Simple Array - False Sharing -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

auto sums = std::vector<my_type>(num_threads);

#pragma omp parallel 
{
const auto id = omp_get_thread_num();

for (int i = (id / num_threads) * n; i < ((id + 1) / num_threads) * n; ++i)
for (int j = 0; j < n; ++j)
sums[id] += matrix[i][j];
}


for (auto s : sums) sum += s;

avg = sum / (n * n);
const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}

#pragma endregion

std::cout << "---------------------------" << std::endl;

#pragma region Parallel For Reduction Static Schedule

{
std::cout << "Parallel For Reduction Static (512) -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

#pragma omp parallel for reduction(+: sum) schedule(static, 512)
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
sum += matrix[i][j];

avg = sum / (n * n);
const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}

#pragma endregion

std::cout << "---------------------------" << std::endl;

#pragma region Parallel For Reduction Dynamic Schedule

{
std::cout << "Parallel For Reduction Dynamic -		";

double sum_duration = 0;

for (auto r = 0; r < num_runs; ++r)
{
const auto t0 = std::chrono::high_resolution_clock::now();

sum = 0;
avg = 0;

#pragma omp parallel for reduction(+: sum) schedule(dynamic)
for (int i = 0; i < n; ++i)
for (int j = 0; j < n; ++j)
sum += matrix[i][j];

avg = sum / (n * n);
const auto t1 = std::chrono::high_resolution_clock::now();
sum_duration += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

const auto duration_ms = (sum_duration / num_runs) / 1000.0f;
std::cout << "Average Duration =  " << duration_ms << "(ms) - ";

std::cout << "Avg =  " << avg;
std::cout << std::endl;
}

#pragma endregion

std::cout << "---------------------------" << std::endl << std::endl;

std::cout << "Max Threads Num = " << omp_get_max_threads() << std::endl << std::endl;
std::cout << "Number of Threads = " << num_threads << std::endl << std::endl;
std::cout << "2D Array Size = " << n << " * " << n << std::endl << std::endl;
std::cout << "Number Of Each Method Run = " << num_runs << std::endl << std::endl;

std::cout << "---------------------------" << std::endl;

std::cout << "Profiling Finished Successfully";

free(matrix);
std::cin >> sum;

return 0;
}