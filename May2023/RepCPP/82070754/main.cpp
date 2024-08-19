#include <iostream>
#include <ctime>
#include <omp.h>
#include <cmath>
#include <zconf.h>
#include "sort.h"

using namespace std;

const int SIZE = 10;

void print_array(int arr[]) {
for (int i = 0; i < SIZE; i++) {
cout << arr[i] << " ";
}
}

int main() {

int size, rank;
int k = 5;



int sum;
#pragma omp parallel for reduction(+:sum) 
for (int i = 0; i < 100; i++) {
rank = omp_get_thread_num();
printf("iteration - %d, thread - %d\n", i, rank);
}

cout << "last - " << rank << endl;
}

