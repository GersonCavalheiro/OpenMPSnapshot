#include <bits/stdc++.h>
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <omp.h>

using namespace std;
using namespace std::chrono;
typedef int32_t Number;
const int sliceSize = 128*1024;

int eratosthenesOddSingleBlock(const int from, const int to)
{
const int memorySize = (to - from + 1) / 2;
char* isPrime = new char[memorySize];
for (int i = 0; i < memorySize; i++)
isPrime[i] = 1;
for (int i = 3; i*i <= to; i+=2)
{
if (i >= 3*3 && i % 3 == 0)
continue;
if (i >= 5*5 && i % 5 == 0)
continue;
if (i >= 7*7 && i % 7 == 0)
continue;
if (i >= 11*11 && i % 11 == 0)
continue;
if (i >= 13*13 && i % 13 == 0)
continue;
int minJ = ((from+i-1)/i)*i;
if (minJ < i*i)
minJ = i*i;
if ((minJ & 1) == 0)
minJ += i;
for (int j = minJ; j <= to; j += 2*i)
{
int index = j - from;
isPrime[index/2] = 0;
}
}
int found = 0;
for (int i = 0; i < memorySize; i++)
found += isPrime[i];
if (from <= 2)
found++;
delete[] isPrime;
return found;
}

int eratosthenes(Number lastNumber, Number sliceSize)
{
omp_set_num_threads(omp_get_num_procs());
int found = 0;
#pragma omp parallel for reduction(+:found)
for (Number from = 2; from <= lastNumber; from += sliceSize)
{
Number to = from + sliceSize;
if (to > lastNumber)
to = lastNumber;
found += eratosthenesOddSingleBlock(from, to);
}
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
eratosthenes(n, 2*sliceSize);
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout << duration.count();
return 0;
}
