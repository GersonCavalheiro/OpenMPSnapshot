#include "pch.h"
#include <iostream>
#include <cstdio>
#include <omp.h>

using namespace std;

int main()
{
int a = 3;
int b[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
int c[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
int d[10];

cout << "1 - " << a << endl;

omp_set_num_threads(3);

#pragma omp parallel private(a) shared(b, c, d)
{
#pragma omp for schedule(dynamic, 2) nowait
for (int i = 0; i < 10; i++) {
d[i] = b[i] + c[i];
printf("%d-%d\n", i, omp_get_thread_num());
}

cout << "fin for\n";
} 








return 0;
}
