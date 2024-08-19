#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <ctime>

using namespace std;

int main(int argc, char* argv[]) {
int N = atoi(argv[1]);
int* v = new int[N];

clock_t start, finish;

for (long unsigned i = 0; i < N; i++)
v[i] = rand() % 1000;

int maxvalue = v[0];

for (int threads = 1; threads <= 10; threads++) {
start = clock();

#pragma omp parallel for reduction(max:maxvalue) num_threads(threads)
for (long unsigned i = 0; i < N; i++) 
maxvalue = (maxvalue > v[i]) ? maxvalue : v[i];


finish = clock();

double time = (double(finish - start) / CLOCKS_PER_SEC);

cout << "# of threads: " << threads << endl;
cout << "Execution time: " << time << endl;
}

delete[] v;

return 0;
}