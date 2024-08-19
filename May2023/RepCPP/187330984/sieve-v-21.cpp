#include <bits/stdc++.h>
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;
typedef int32_t Number;

int eratosthenes(int lastNumber)
{
omp_set_num_threads(omp_get_num_procs());
const Number lastNumberSqrt = (int)sqrt((double)lastNumber);
Number memorySize = (lastNumber-1)/2;
char* isPrime = new char[memorySize+1];
#pragma omp parallel for
for (Number i = 0; i <= memorySize; i++)
isPrime[i] = 1;

#pragma omp parallel for schedule(auto)
for (Number i = 3; i <= lastNumberSqrt; i += 2)
if (isPrime[i/2])
for (Number j = i*i; j <= lastNumber; j += 2*i)
isPrime[j/2] = 0;

int found = lastNumber >= 2 ? 1 : 0;
#pragma omp parallel for
for (Number i = 1; i <= memorySize; i++)
found += isPrime[i];

delete[] isPrime;
return found;
}

int main(int argc, char *argv[])
{
int n = 100;
if (argc > 1)
{
try
{
n = stoi(argv[1]);
}
catch (const std::exception &e)
{
std::cerr << e.what() << '\n';
}
}
auto start = high_resolution_clock::now();
eratosthenes(n);
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << duration.count();
return 0;
}
