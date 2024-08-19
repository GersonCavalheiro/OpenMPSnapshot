#pragma GCC optimize("03", "unroll-loops", "omit-frame-pointer","inline")
#pragma GCC option("arch=native" ,"tune-native", "n0-zero-upper")
#pragma GCC target("avx")

#include<bits/stdc++.h>
#include<random>

using namespace std;
using namespace std::chrono;

int main()
{
int size = 1000000;
float arr[size];
std::random_device randomDevice;
std::mt19937 mt19937(randomDevice());
std::normal_distribution<float> distribution(0.0, 100000);
for(int i =0; i <size; i++)
{
arr[i] = distribution(mt19937);
}
auto time = high_resolution_clock:: now();
for(int i =0; i<size; i++)
{
for(int j =0;j<size-i-1;j++)
{
if(arr[j] > arr[j+1])
{
float temp = arr[j];
arr[j] = arr[j+1];
arr[j+1] = temp;
}
}
}
auto end_time = duration_cast<duration<double>>(high_resolution_clock::now() - time).count();
cout << "Time used: "<< end_time << endl;
for(int i =0; i < size-1; i++)
{
if(arr[i] > arr[i+1])
{
cout<<"  FAILED  ";
}
}
return 0;
}