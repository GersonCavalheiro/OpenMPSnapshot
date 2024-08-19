#include "iostream"
#include "omp.h"
#include "../../array_utils.h"

using namespace std;

const int ARRAY_SIZE = 12;

int main() {

int *a = new int[ARRAY_SIZE];
int *b = new int[ARRAY_SIZE];
int *c = new int[ARRAY_SIZE];

#pragma omp parallel for schedule(static) num_threads(3)
for (int i = 0; i < ARRAY_SIZE; i++) {
int val1 = i + 1;
a[i] = val1;
printf("Thread id: %d, from threads: %d: generated values is %d in [[a]] array\n",
omp_get_thread_num(), omp_get_num_threads(), val1);
int val2 = 12 - i;
b[i] = val2;
printf("Thread id: %d, from threads: %d: generated values is %d in [[b]] array\n",
omp_get_thread_num(), omp_get_num_threads(), val2);
}
print_array(a, ARRAY_SIZE);
cout << "\n";
print_array(b, ARRAY_SIZE);
cout << "\n";
#pragma omp parallel for schedule(dynamic, 2) num_threads(4)
for (int i = 0; i < ARRAY_SIZE; i++) {
int val = b[i] + a[i];
c[i] = val;
printf("Thread id: %d, from threads: %d: generated values is %d in [[c]] array\n",
omp_get_thread_num(), omp_get_num_threads(), val);
}
print_array(c, ARRAY_SIZE);
delete[] a, b, c;
}

