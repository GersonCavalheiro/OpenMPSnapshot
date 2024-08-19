#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>

#include <benchmark/benchmark.h>

template <typename T, size_t Size = 4096>
class Laplace2D : public benchmark::Fixture {

public:

std::vector<T> A;
std::vector<T> Anew;

int n {Size};
int m;

int iter_max = 100;
T tol {1.0e-6};

Laplace2D() : A(Size*Size), Anew(Size*Size), m {n} {};

};


BENCHMARK_TEMPLATE_DEFINE_F(Laplace2D, Double_4096, double, 4096) (benchmark::State& state) {

double error;
int iter;


for (auto _ : state) {

error = 1.0;
iter = 0;

for (size_t j = 0; j < n; j++)
{
A[j*n]    = 1.0;
Anew[j*n] = 1.0;
}

auto nthreads = state.range(0);

auto begin = omp_get_wtime();

while ( error > tol && iter < iter_max )
{
error = 0.0;

#pragma omp parallel num_threads(nthreads) default(none) shared(Anew, A, n, m, error)
{

#pragma omp for collapse(2) reduction(max:error) schedule(dynamic,256)
for (size_t j = 1; j < n - 1; j++) {
for (size_t i = 1; i < m - 1; i++) {
Anew[j*n+i] = 0.25 * (A[j*n + i + 1] + A[j*n + i - 1]
+ A[(j - 1)*n + i] + A[(j + 1)*n + i]);
error = fmax(error, fabs(Anew[j*n+i] - A[j*n+i]));
}
}

#pragma omp for collapse(2) schedule(dynamic,256)
for (size_t j = 1; j < n - 1; j++) {
for (size_t i = 1; i < m - 1; i++) {
A[j*n+i] = Anew[j*n+i];
}
}
}

iter++;

if(iter % 10 == 0) printf("%5d, %0.8lf\n", iter, error);
}

auto time = omp_get_wtime() - begin;

state.SetIterationTime( time );

}

}
BENCHMARK_REGISTER_F(Laplace2D, Double_4096)
->Arg(1)->Arg(2)->Arg(4)->Arg(8)
->UseManualTime();
