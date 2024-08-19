#include <omp.h>
#include <iostream>
#include "stdio.h"
#include "stdlib.h"

using namespace std;

int main(int argc, char **argv) {
const int size = (argc > 1 ? atoi(argv[1]) : 1000);
int *a = new int[size];
#pragma omp parallel for
for (int i = 0; i < size; i++) {
a[i] = i;
}

int sum = 0;
#pragma omp parallel for
for (int i = 0; i < size; i++) {
#pragma omp critical
sum += a[i];
}
cout << "Sum = " << sum << endl;
delete[] a;
return 0;
}