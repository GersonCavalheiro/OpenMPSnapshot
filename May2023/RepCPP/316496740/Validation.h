

#pragma once

#include "Common.h"

void export_multidimentional_float_vector(const std::vector<std::array<float, Nv>> Obj, std::string filename);
void export_multidimentional_integer_array(const std::array<std::vector<int>, Nc> Obj, std::string filename);
void track_kmeans_progress(int iter_counter, long double norm_iter_conv);
void export_kmeans_progress(std::string filename);