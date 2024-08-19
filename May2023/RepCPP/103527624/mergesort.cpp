#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <chrono>

#ifdef __cplusplus
extern "C" {
#endif

void generateMergeSortData (int* arr, size_t n);
void checkMergeSortResult (int* arr, size_t n);


#ifdef __cplusplus
}
#endif
using namespace std;

void mergesort(int *a, int n, int *temp);
void merge(int *a, int n, int *temp);
int main (int argc, char* argv[]) {

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

if (argc < 3) {
std::cerr<<"Usage: "<<argv[0]<<" <n> <nbthreads>"<<std::endl;
return -1;
}

int n = atoi(argv[1]);
int nbthreads = atoi(argv[2]);
int *temp = new int[n];
int * arr = new int [n];

generateMergeSortData (arr, atoi(argv[1]));

omp_set_num_threads(nbthreads);

std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
#pragma omp parallel
{
#pragma omp single
mergesort(arr, n, temp);
}

std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
std::chrono::duration<double> elapsed_seconds = end-start;
std::cerr<<elapsed_seconds.count()<<std::endl;

checkMergeSortResult (arr, atoi(argv[1]));

delete[] arr;

return 0;
}
void merge(int *a, int n, int *temp)
{
int i = 0, j = n/2, k = 0;

while (i<n/2 && j<n)
{
if (a[i] < a[j])
{
temp[k++] = a[i++];
}
else
{
temp[k++] = a[j++];
}
}
while (i<n/2)
{
temp[k++] = a[i++];
}

while (j<n)
{
temp[k++] = a[j++];
}
memcpy(a, temp, n*sizeof(int));
return;
}
void mergesort(int *a, int n, int *temp)
{
if (n < 2)
return;

#pragma omp task
mergesort(a, n/2, temp);
#pragma omp taskwait

#pragma omp task
mergesort(a+(n/2), n-(n/2), temp);

#pragma omp taskwait

merge(a, n, temp);
}
