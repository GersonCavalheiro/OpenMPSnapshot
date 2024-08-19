#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

void generatePrefixSumData (int* arr, size_t n);
void checkPrefixSumResult (int* ans, size_t n);


#ifdef __cplusplus
}
#endif


int* calcPrefixSum(int* arr, int* pr, int n, int nbthreads);
int main (int argc, char* argv[]) {
int nbthreads, n;
if (argc < 3) {
std::cerr<<"Usage: "<<argv[0]<<" <n> <nbthreads>"<<std::endl;
return -1;
}

nbthreads = atoi(argv[2]);
n = atoi(argv[1]);


#pragma omp parallel
{
int fd = open (argv[0], O_RDONLY);
if (fd != -1) {
close (fd);
}
else {
std::cerr<<"something is amiss"<<std::endl;
}
}

int * arr = new int [atoi(argv[1])];
int * pr = new int [atoi(argv[1])+1];
generatePrefixSumData (arr, atoi(argv[1]));
omp_set_num_threads(nbthreads);

std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
int* ans = calcPrefixSum(arr, pr, n, nbthreads);
std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed_seconds = end-start;
std::cerr<<elapsed_seconds.count()<<std::endl;

checkPrefixSumResult(ans, atoi(argv[1]));

delete[] arr;

return 0;
}
int* calcPrefixSum(int* arr, int* pr, int n, int nbthreads){
int* blocks = new int[nbthreads];
pr[0]=0;
#pragma omp parallel
{
int threadNum = omp_get_thread_num();
#pragma omp single
{
blocks = new int[nbthreads+1];
blocks[0]=0;
}
int sum=0;
#pragma omp for schedule(static)
for(int i=0; i<n; i++){
sum += arr[i];
pr[i+1] = sum;
}

blocks[threadNum+1] = sum;
#pragma omp barrier
int offset=0;
for(int i=0; i<(threadNum+1); i++){
offset+=blocks[i];
}	

#pragma omp for schedule(static)
for(int i=0; i<n; i++){
pr[i+1] += offset;
}

}
delete[] blocks;
return pr;
}
