#include <omp.h>
#include <iostream>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define MAX_NUMBER 20

#ifdef __cplusplus
extern "C" {
#endif

void generateReduceData (int* arr, size_t n);

#ifdef __cplusplus
}
#endif

using namespace std;

int sum=0;

void reduce(int index, int * arr){
if(index>=0){
#pragma omp task
{
reduce(index-1,arr);
}
sum+=arr[index];
}
#pragma omp taskwait 
}

int main(int argc,char* argv[]){

int n = stoi(argv[1]);
int threads = stoi(argv[2]);

omp_set_num_threads(threads);

std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

int * arr = new int [n];

generateReduceData (arr, n);

#pragma omp parallel
{
#pragma omp single
{
#pragma omp task
reduce(n-1,arr);
}
}
#pragma omp taskwait

std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

cout<<sum<<endl;

std::chrono::duration<double> elapsed_seconds = end-start;

std::cerr<<elapsed_seconds.count()<<std::endl;

return 0;

}

