

#pragma once

#include "Common.h"

std::chrono::time_point<std::chrono::system_clock> init_benchmark(void);
std::chrono::time_point<std::chrono::system_clock> bench_convergence(void);
std::chrono::duration<double> bench_loop(std::chrono::time_point<std::chrono::system_clock>, std::chrono::time_point<std::chrono::system_clock>);
std::chrono::time_point<std::chrono::system_clock> terminate_bench(void);
std::pair<std::chrono::duration<double>, std::chrono::duration<double>> benchmark_results(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> loop_benchmark, std::chrono::time_point<std::chrono::system_clock> end);