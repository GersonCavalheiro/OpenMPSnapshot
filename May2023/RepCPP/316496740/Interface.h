

#pragma once

#include "Common.h"

void multidimentional_float_vector_interface(const std::vector<std::array<float, Nv>>& Obj, std::string Obj_name);
void multidimentional_int_array_interface(const std::array<std::vector<int>, Nc>& Obj, std::string Obj_name);
void progress_interface(const int iter_counter, const long double iter_conv, const long double norm_iter_conv, const std::chrono::duration<double> loop_benchmark);
void kmeans_progress(const int iter_counter, const long double iter_conv, const long double norm_iter_conv, const std::chrono::duration<double> loop_benchmark, const int verbose);
void kmeans_termination(const std::pair<std::chrono::duration<double>, std::chrono::duration<double>> bench_results, const long double iter_conv);