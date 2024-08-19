#include <iostream>
#include <omp.h>

using namespace std;

#define NUM_THREADS 4

int fact(int n);

int main() {

int num; 
cin >> num;

float start_time = omp_get_wtime();

omp_set_num_threads(NUM_THREADS);

int fact_result = 0;

#pragma omp parallel
{
#pragma omp single nowait
{
fact_result = fact(num);
}
}

cout << "Factorial of " << num << ": " << fact_result << endl;

float elapsed_time = omp_get_wtime() - start_time;

cout << "Elapsed time is: " << elapsed_time << endl;
return 0;
}

int fact(int n) {

if(n == 0)
return 1;

int b = 0;

#pragma omp task shared(b)
b = fact(n - 1);

#pragma omp taskwait
return n * b;

}