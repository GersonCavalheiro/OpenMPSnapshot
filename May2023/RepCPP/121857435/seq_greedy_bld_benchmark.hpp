#pragma once



#include <celero/Celero.h>

#include "bench_utils.hpp"
#include "bnmf_algs.hpp"
#include <fstream>
#include <iostream>

namespace seq_greedy_bld_bench_vars {
long x = 100;
long y = 100;
double beg = 0;
double scale = 5;
size_t r = 17;
} 

BASELINE(seq_greedy_bld, 100x100, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 100;
y = 100;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(seq_greedy_bld, 200x100, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 200;
y = 100;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(seq_greedy_bld, 300x100, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 300;
y = 100;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(seq_greedy_bld, 400x100, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 400;
y = 100;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(seq_greedy_bld, 100x200, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 100;
y = 200;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(seq_greedy_bld, 100x300, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 100;
y = 300;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}

BENCHMARK(seq_greedy_bld, 100x400, 30, 1) {
using namespace bnmf_algs;
using namespace benchmark;
using namespace seq_greedy_bld_bench_vars;

x = 100;
y = 400;

matrixd X = make_matrix(x, y, beg, scale);
auto params = make_params<double>(x, y, r);
celero::DoNotOptimizeAway(bld::seq_greedy_bld(X, r, params));
}
