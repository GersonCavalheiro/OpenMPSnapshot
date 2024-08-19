

#include <iostream>
#include <random>
#include <omp.h>
using namespace std;

bool debug = false;


template <typename T>
T sum_vect(T x[], int n, int thread_count)
{
T sum{ };
#pragma omp parallel for num_threads(thread_count) \
reduction(+: sum)
for (int i = 0; i < n; i++)
sum += x[i];
return sum;
}


double *generate_vector(int n)
{
default_random_engine generator;
uniform_real_distribution<double> distribution{ 0.0, 1.0 };

double *x = new double[n];
for (int i = 0; i < n; i++)
x[i] = distribution(generator);
return x;
}


double *read_vector(int n)
{
double *x = new double[n];
for (int i = 0; i < n; i++)
cin >> x[i];
return x;
}


int main(int argc, char *argv[])
{
int thread_count = stoi(argv[1]), n = stoi(argv[2]);

double *x = nullptr;
if (debug)
{
cout << "Enter the vector: " << endl;
x = read_vector(n);
}
else
{
cout << "Generated vector of size " << n << endl;
x = generate_vector(n);
}

double start = omp_get_wtime();
int sum = sum_vect(x, n, thread_count);
double finish = omp_get_wtime(), elapsed = finish - start;
cout << "Sum calculated. Elapsed time: " << elapsed << " seconds" << endl;

if (debug)
{
cout << "The sum is: " << sum << endl;
}

delete[] x;
return 0;
}
