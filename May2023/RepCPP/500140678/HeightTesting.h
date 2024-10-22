#pragma once
#include <vector>
#include <SPH2D_FIO.h>
#include <fmt/format.h>
#include <omp.h>

#include "HeightTestingParams.h"

class HeightTesting {
using Grid = sphfio::Grid;
using TimeLayer = sphfio::TimeLayer;
using ParamsPtr = sphfio::ParamsPtr;

const Grid& grid;
const ParamsPtr params;
public:
HeightTesting(const Grid& grid, ParamsPtr params) :
grid{ grid },
params{ params }
{
}

double maxInLayer(const TimeLayer& layer, double x, double search_n) {
double search_radius = search_n * params->hsml;

std::vector<double> max_values;
#pragma omp critical 
{
max_values = std::vector<double>(omp_get_max_threads());
}

#pragma omp parallel 
{
int thread = omp_get_thread_num();
#pragma omp for
for (int i = 0; i < layer.ntotal; ++i) {
double current_x = layer.r(i).x;
double current_y = layer.r(i).y;

if (std::fabs(current_x - x) < search_radius) {
max_values[thread] = std::max(max_values[thread], current_y);
}
}
}

std::sort(std::begin(max_values), std::end(max_values), std::greater<double>());
return max_values[0];
}

std::vector<double> timeProfile(double x, double search_n) {
std::vector<double> waves_height(grid.size());

#pragma omp parallel for
for (int i = 0; i < grid.size(); ++i) {
waves_height[i] = maxInLayer(grid.at(i), x, search_n);
}
return waves_height;
}

std::vector<double> spaceProfile(double t, double search_n) {
double width = params->x_maxgeom - params->x_mingeom;
int N = static_cast<int>(width / params->delta);
std::vector<double> waves_height(N);

int layer_num = static_cast<int>(t / params->dt);
auto grid_iter = grid.find(layer_num);
if (grid_iter == grid.end()) {
throw std::runtime_error{ fmt::format("error: searching for {} layer of {}", layer_num, grid.size()) };
}
auto& layer = *grid_iter;

#pragma omp parallel for
for (int i = 0; i < N; ++i) {
double x = params->x_mingeom + params->delta * i;
waves_height[i] = maxInLayer(layer, x, search_n);
}
return waves_height;
}


};