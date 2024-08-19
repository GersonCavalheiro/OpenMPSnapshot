#include <iostream>

#include <omp.h>
#include <cmath>
#include <chrono>


const long long max = pow(10,8);


int main() {


auto start = std::chrono::steady_clock::now();
long long sum = 0;
for (long long i = 0 ; i < max;i++){
sum++;
}
for (long long i = 0 ; i < max;i++){
sum--;
}

auto end = std::chrono::steady_clock::now();
std::chrono::duration<double> elapsed_seconds = end-start;
std::cout << "Sequence operation elapsed time: " << elapsed_seconds.count() << "s\n";
printf("Result: %i\n", sum);




start = std::chrono::steady_clock::now();
sum = 0;
#pragma omp parallel shared(sum)
{
#pragma omp sections
{
#pragma omp section
{
for (long long i = 0 ; i < max;i++){
sum++;
}
}
#pragma omp section
{
for (long long i = 0 ; i < max;i++){
sum--;
}
}
}
}
end = std::chrono::steady_clock::now();
elapsed_seconds = end-start;
std::cout << "No access control elapsed time: " << elapsed_seconds.count() << "s\n";
printf("Result: %i\n", sum);


start = std::chrono::steady_clock::now();
sum = 0;
#pragma omp parallel shared(sum)
{
#pragma omp sections
{
#pragma omp section
{
for (long long i = 0 ; i < max;i++){
#pragma omp atomic
sum++;
}
}
#pragma omp section
{
for (long long i = 0 ; i < max;i++){
#pragma omp atomic
sum--;
}
}
}
}
end = std::chrono::steady_clock::now();
elapsed_seconds = end-start;
std::cout << "Access control by using atomic operations elapsed time: " << elapsed_seconds.count() << "s\n";
printf("Result: %i\n", sum);


start = std::chrono::steady_clock::now();
sum = 0;
#pragma omp parallel shared(sum)
{
#pragma omp sections
{
#pragma omp section
{
for (long long i = 0 ; i < max;i++){
#pragma omp critical
{
sum++;
}
}
}
#pragma omp section
{
for (long long i = 0 ; i < max;i++){
#pragma omp critical
{
sum--;
}
}
}
}
}
end = std::chrono::steady_clock::now();
elapsed_seconds = end-start;
std::cout << "Access control by using critical seciton elapsed time: " << elapsed_seconds.count() << "s\n";
printf("Result: %i\n", sum);



start = std::chrono::steady_clock::now();
sum = 0;
#pragma omp parallel shared(sum)
{
#pragma omp sections
{
#pragma omp section
{
long long tmpSum = 0;
#pragma omp parallel for reduction(+:tmpSum)
for (long long i = 0 ; i < max;i++){
tmpSum++;
}
#pragma omp atomic
sum += tmpSum;
}
#pragma omp section
{
long long tmpSum = 0;
#pragma omp parallel for reduction(+:tmpSum)
for (long long i = 0 ; i < max;i++){
tmpSum--;
}
#pragma omp atomic
sum += tmpSum;
}
}
}
end = std::chrono::steady_clock::now();
elapsed_seconds = end-start;
std::cout << "Nested loops elapsed time: " << elapsed_seconds.count() << "s\n";
printf("Result: %i\n", sum);



start = std::chrono::steady_clock::now();
sum = 0;
long long tmpSumPlus = 0;
long long tmpSumMinus = 0;
#pragma omp parallel shared(sum, tmpSumPlus, tmpSumMinus)
{
#pragma omp for nowait reduction(+:tmpSumPlus)
for (long long i = 0 ; i < max;i++){
tmpSumPlus++;
}
#pragma omp for nowait reduction(+:tmpSumMinus)
for (long long i = 0 ; i < max;i++){
tmpSumMinus--;
}
#pragma omp barrier
#pragma omp single
{
sum += tmpSumPlus;
sum += tmpSumMinus;
}
}
end = std::chrono::steady_clock::now();
elapsed_seconds = end-start;
std::cout << "Sequential paralelized for loops: " << elapsed_seconds.count() << "s\n";
printf("Result: %i\n", sum);



return 0;
}
