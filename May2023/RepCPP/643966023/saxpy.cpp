#include <iostream>
#include <omp.h>

#include <benchmark/benchmark.h>

template <typename T, size_t N = 256>
class Saxpy : public benchmark::Fixture {

public:

T scalar {2.};

std::size_t size;
std::vector<T> x;
std::vector<T> y;

Saxpy() : x(N, 0.), y(N, 0.), size {N} {};

void SetUp(const ::benchmark::State& state) override {

std::fill(x.begin(), x.end(), 1.);
std::fill(y.begin(), y.end(), 2.);

}

};


BENCHMARK_TEMPLATE_DEFINE_F(Saxpy, Double_256, double, 256) (benchmark::State& state) {

for (auto _ : state) {

auto nthreads = state.range(0);

auto begin = omp_get_wtime();

#pragma omp parallel for num_threads(nthreads) default(none) shared(x, y, scalar, size)
for (std::size_t i=0; i<size; i++ )
{
y[i] = scalar * x[i] + y[i];
}

auto time = omp_get_wtime() - begin;

state.SetIterationTime( time );

}

}
BENCHMARK_REGISTER_F(Saxpy, Double_256)
->Arg(1)->Arg(2)->Arg(4)->Arg(8)
->UseManualTime();


BENCHMARK_TEMPLATE_DEFINE_F(Saxpy, Double_512_3, double, 512*512*512) (benchmark::State& state) {

for (auto _ : state) {

auto nthreads = state.range(0);

auto begin = omp_get_wtime();

#pragma omp parallel for num_threads(nthreads) default(none) shared(x, y, scalar, size)
for (std::size_t i=0; i<size; i++ )
{
y[i] = scalar * x[i] + y[i];
}

auto time = omp_get_wtime() - begin;

state.SetIterationTime( time );

}

}
BENCHMARK_REGISTER_F(Saxpy, Double_512_3)
->Arg(1)->Arg(2)->Arg(4)->Arg(8)
->UseManualTime();
