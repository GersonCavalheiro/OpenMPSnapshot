#include <iostream>
#include <omp.h>

#include <benchmark/benchmark.h>

#define PI25DT 3.141592653589793238462643

template <typename T, size_t Intervals = 100000000>
class Pi_Value : public benchmark::Fixture {

public:

long int intervals = Intervals;

double pi;

};


BENCHMARK_TEMPLATE_DEFINE_F(Pi_Value, Double_1E5, double, 10000) (benchmark::State& state) {

double x, f, sum, dx;


for (auto _ : state) {

sum = 0.0;
dx = 1.0 / (double) intervals;

auto nthreads = state.range(0);

auto begin = omp_get_wtime();

#pragma omp parallel for num_threads(nthreads) default(none) private(x,f) shared(intervals, dx) \
reduction(+:sum)
for (std::size_t i = 1; i <= intervals; i++) {
x = dx * ((double) i - 0.5);
f = 4.0 / (1.0 + x*x);
sum = sum + f;
}

pi = dx*sum;

auto time = omp_get_wtime() - begin;

state.SetIterationTime( time );

}

}
BENCHMARK_REGISTER_F(Pi_Value, Double_1E5)
->Arg(1)->Arg(2)->Arg(4)->Arg(8)
->UseManualTime();




BENCHMARK_TEMPLATE_DEFINE_F(Pi_Value, Double_1E8, double, 100000000) (benchmark::State& state) {

double x, f, sum, dx;

for (auto _ : state) {

sum = 0.0;
dx = 1.0 / (double) intervals;

auto nthreads = state.range(0);

auto begin = omp_get_wtime();

#pragma omp parallel for num_threads(nthreads) default(none) private(x,f) shared(intervals, dx) \
reduction(+:sum)
for (std::size_t i = 1; i <= intervals; i++) {
x = dx * ((double) i - 0.5);
f = 4.0 / (1.0 + x*x);
sum = sum + f;
}

pi = dx*sum;

auto time = omp_get_wtime() - begin;

state.SetIterationTime( time );

}

}

BENCHMARK_REGISTER_F(Pi_Value, Double_1E8)
->Arg(1)->Arg(2)->Arg(4)->Arg(8)
->UseManualTime();
