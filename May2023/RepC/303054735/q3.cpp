#include <iostream>
#include "omp.h"
#include <vector>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define LEN    65536    
#define TMAX   20       
#define ITERATIONS 100  
vector<float> X (LEN, 1.1);
vector<float> Y (LEN, 2.1);
int a = 2;
void operation(int num_t)
{
omp_set_num_threads(num_t);
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < LEN; ++i)
{
X[i] = a*X[i] + Y[i];
}
}
}
int calcAvgTime(int n)
{
X = vector<float> (LEN, 1.1);
Y = vector<float> (LEN, 2.1);
high_resolution_clock::time_point t1 = high_resolution_clock::now();
operation(n);
high_resolution_clock::time_point t2 = high_resolution_clock::now();
auto duration = duration_cast<microseconds>( t2 - t1 ).count();
return (int)duration;
}
int main () {
int avg_times[TMAX+1];
for (int i = 1; i <= 1; ++i)
{
avg_times[i] = calcAvgTime(i);
}
for (int i = 1; i <= TMAX; ++i)
{
avg_times[i] = 0;
}
for (int x = 0; x < ITERATIONS; ++x)
{
for (int i = 1; i <= TMAX; ++i)
{
avg_times[i] += calcAvgTime(i);
}
}
for (int i = 1; i <= TMAX; ++i)
{
avg_times[i] /= ITERATIONS;
}
cout<<"Num_threads\tExec Time (micro-s)"<<endl;
for (int i = 1; i <= TMAX; ++i)
{
cout<<i<<"\t      "<<avg_times[i]<<endl;
}
return 0;
}
