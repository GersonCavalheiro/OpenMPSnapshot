#include <iostream>
#include <stdio.h>
#include <omp.h>

using namespace std;

#define MAX_THREADS 4

int fib(int n);

int main() {

int num;

cout << "Please enter a number: " << endl;
cin >> num;

float time_start = omp_get_wtime();

omp_set_num_threads(MAX_THREADS);

cout << "Fibonacci final result: " << endl << fib(num) << endl;

float elapsed_time = omp_get_wtime() - time_start;
cout << "Computation time: " << elapsed_time << endl;

system("pause");

return 0;
}

int fib(int n) {

int a, b;

if(n == 1 || n == 0)
return 1;


#pragma omp parallel
{
#pragma omp single nowait
{
#pragma omp task
a = fib(n - 1);

#pragma omp task
b = fib(n - 2);
}

}

return a + b;
}