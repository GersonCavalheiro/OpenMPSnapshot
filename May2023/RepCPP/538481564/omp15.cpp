#include <algorithm>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <vector>

using namespace std;

int main() {

omp_set_dynamic(0);
omp_set_num_threads(8);

int left_border, right_border;
cout << "Enter min and max values: ";
cin >> left_border >> right_border;

vector<int> result;
#pragma omp parallel for
for(int i = left_border; i <= right_border; i++) {
if(i >= 2) {
bool flag = true;
#pragma omp parallel for
for(int j = 2; j <= (int) sqrt(i); j++) {
if (i % j == 0) {
flag = false;
}
}
if (flag) {
#pragma omp critical
result.push_back(i);
}
}
}

sort(result.begin(), result.end());

printf("Prime numbers: ");
for(int num : result) {
printf("%d ", num);
}
printf("\n");
}


































