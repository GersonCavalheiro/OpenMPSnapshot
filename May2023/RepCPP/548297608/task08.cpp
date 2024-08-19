
#include <iostream>
#include <ctime>
#include <windows.h>
#include <omp.h>

void initArrays(int *a, int *b, int m, int n) {
for (int i = 0; i < m; i++) {
for (int j = 0; j < n; j++) {
a[i * n + j] = std::rand() % 100;
}
}
for (int j = 0; j < n; j++) {
b[j] = std::rand() % 100;
}
}

void free(int *c, int m) {
for (int i = 0; i < m; i++) {
c[i] = 0;
}
}

void printArray(int *c, int m) {
for (int i = 0; i < m; i++) {
std::cout << std::to_string(c[i]) + " ";
}
std::cout << "\n\n";
}

void runNoParallel(const int *a, const int *b, int *c, int m, int n) {
for (int i = 0; i < m; i++) {
c[i] = 0;
for (int j = 0; j < n; j++) {
c[i] += a[i * n + j] * b[j];
}
}
}

void runParallel(const int *a, const int *b, int *c, int m, int n) {
#pragma omp parallel
{
int nThreads = omp_get_num_threads();
int threadId = omp_get_thread_num();
int itemsPerThread = m / nThreads;
int l = threadId * itemsPerThread;
int r = (threadId == nThreads - 1) ? (m) : (l + itemsPerThread);
for (int i = l; i < r; i++) {
c[i] = 0;
for (int j = 0; j < n; j++) {
c[i] += a[i * n + j] * b[j];
}
}
}
}

const int m = 15000, n = 15000;
int a[m * n], b[n], c[m];

int main() {
std::srand(std::time(nullptr));
DWORD t;
DWORD noParallel;
DWORD parallel;

initArrays(a, b, m, n);

t = GetTickCount();
runNoParallel(a, b, c, m, n);
noParallel = GetTickCount() - t;
std::cout << "Result time (NO PARALLEL) = " << noParallel << " ms.\n";

free(c, m);

t = GetTickCount();
runParallel(a, b, c, m, n);
parallel = GetTickCount() - t;
std::cout << "Result time (PARALLEL) = " << parallel << " ms.\n";

if (noParallel < parallel)
std::cout << "No parallel algorithm was faster.";
else if (noParallel > parallel)
std::cout << "Parallel algorithm was faster.";
else
std::cout << "The results are the same.";

return 0;
}
