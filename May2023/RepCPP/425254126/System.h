#pragma once

#include <vector>
#include <utility>
#include <algo/interfaces/parallel/ParallelSweepMethod.h>

#include "task/interfaces/constants/AppConstants.h"
#include "algo/interfaces/serial/SerialSweepMethod.h"

template<class T>
using vec3 = std::vector<T>;
template<class T>
using matr2d = std::vector<std::vector<T>>;
template<class T>
using vec2d = vec3<matr2d<T>>;
template<class T>
using matr3d = std::vector<std::vector<std::vector<T>>>;
template<class T>
using vec3d = vec3<matr3d<T>>;

#define loop3(x) for (size_t x = 0; (x) < 3; (x)++)
#define loop(x, n) for (size_t x = 0; (x) < (n); (x)++)

#include "task/interfaces/tools/Parameters.h"
#include "task/interfaces/tools/Area.h"
#include "task/interfaces/tools/Grid.h"
#include "InitConditions.h"
#include "fstreams/interfaces/FWork.h"

class System {
private:
void assignAllU() {
matr2d<double> null(grid[0], vec(grid[1], 0.));

uLow.assign(3, null);
uMid.assign(3, null);
uTop.assign(3, null);
}

pairs kappa, mu, gamma;

protected:
void executeSerialSweepLayersPhase1() {
size_t n1 = grid[0];
size_t n2 = grid[1];

loop3(k) {
vec a(n2 - 2, A[k]);
vec c(n2 - 2, C[k]);
vec b(n2 - 2, B[k]);
vec phi(n2 - 2, 0.);

for (size_t i = 1; i < n1 - 1; i++) {
for (size_t j = 1; j < n2 - 1; j++) {
phi[j - 1] = Phi[k][i][j];
}

SerialSweepMethod ssm(a, c, b, phi, kappa, mu, gamma);
vec u = ssm.run();

for (size_t j = 0; j < n2; j++) {
uMid[k][i][j] = u[j];
}
}
}
}

void executeSerialSweepLayersPhase2() {
size_t n1 = grid[0];
size_t n2 = grid[1];

loop3(k) {
vec a(n1 - 2, A[k]);
vec c(n1 - 2, C[k]);
vec b(n1 - 2, B[k]);
vec phi(n1 - 2, 0.);

for (size_t j = 1; j < n2 - 1; j++) {
for (size_t i = 1; i < n1 - 1; i++) {
phi[i - 1] = Phi[k][i][j];
}

SerialSweepMethod ssm(a, c, b, phi, kappa, mu, gamma);
vec u = ssm.run();

for (size_t i = 0; i < n1; i++) {
uTop[k][i][j] = u[i];
}
}
}
}

void executeParallelSweepLayersPhase1() {
size_t n1 = grid[0];
size_t n2 = grid[1];

loop3(k) {
vec a(n2 - 2, A[k]);
vec c(n2 - 2, C[k]);
vec b(n2 - 2, B[k]);
vec phi(n2 - 2, 0.);

for (size_t i = 1; i < n1 - 1; i++) {
for (size_t j = 1; j < n2 - 1; j++) {
phi[j - 1] = Phi[k][i][j];
}

ParallelSweepMethod ssm(n1, a, c, b, phi, kappa, mu, gamma, 2);
vec u = ssm.run();

#pragma omp barrier
for (size_t j = 0; j < n2; j++) {
uMid[k][i][j] = u[j];
}
}
}
}

void executeParallelSweepLayersPhase2() {
size_t n1 = grid[0];
size_t n2 = grid[1];

loop3(k) {
vec a(n1 - 2, A[k]);
vec c(n1 - 2, C[k]);
vec b(n1 - 2, B[k]);
vec phi(n1 - 2, 0.);

for (size_t j = 1; j < n2 - 1; j++) {
for (size_t i = 1; i < n1 - 1; i++) {
phi[i - 1] = Phi[k][i][j];
}

ParallelSweepMethod ssm(n2, a, c, b, phi, kappa, mu, gamma, 2);
vec u = ssm.run();

#pragma omp barrier
for (size_t i = 0; i < n1; i++) {
uTop[k][i][j] = u[i];
}
}
}
}

public:
vec3<pairs> area;

vec3<int> grid;

vec3<pairs> params;

vec3<double> step;

vec3<vec> nodes;


vec2d<double> uLow;
vec2d<double> uMid;
vec2d<double> uTop;

vec3<double> lambda;


vec3<double> A, B, C;

vec2d<double> Phi;

vec2d<double> F;

static bool printVec2d(const vec2d<double>& r, const str& name) {
loop3(k) {
Instrumental::printMatr(r[k], name + std::to_string(k));
}

return true;
}

System() = default;

System(const Area& area_, const Grid& grid_, Parameters params_,
const vec3<double>& lambda_)
: area(area_.getData()), grid(grid_.getData()), params(params_.getData()),
lambda(lambda_)
{
this->assignAllU();
this->defineStep();
this->defineNodes();

Phi.assign(3, matr2d<double>(grid[0], vec(grid[1], 0.)));
}

void defineStep() {
step.resize(3);
loop3(i) {
step[i] = (area[i].second - area[i].first) / (grid[i] - 1);
}
}

void defineNodes() {
nodes.resize(3);
loop3(k) {
nodes[k].assign(grid[k], 0.);

loop(ijs, grid[k]) {
nodes[k][ijs] = area[k].first + (double)ijs * step[k];
}
}
}

void defineF(size_t phase) {
F = (phase == 0)
? this->declareF(uLow)
: this->declareF(uMid);
}

vec2d<double> declareF(const vec2d<double>& u) {
vec2d<double> res(3, matr2d<double>(grid[0], vec(grid[1], 0.)));

for (int i = 0; i < grid[0]; i++) {
for (int j = 0; j < grid[1]; j++) {
res[0][i][j] = u[0][i][j] * (params[0].first - u[0][i][j] - params[1].first * u[1][i][j] - params[2].first * u[2][i][j]);
res[1][i][j] = u[1][i][j] * (params[0].second - params[1].second * u[0][i][j] - u[1][i][j] - params[2].second * u[2][i][j]);
res[2][i][j] = u[2][i][j] * (-1. + params[3].first * u[0][i][j] + params[3].second * u[1][i][j]);
}
}

return res;
}

void defineLayersParams(size_t phase) {
A.assign(3, 0.);
B.assign(3, 0.);
C.assign(3, 0.);

loop3(k) {
auto common = [&](double h12) {
return step[2] * lambda[k] / std::pow(h12, 2);
};

auto formulaAB = [&](double h12) {
return -0.5 * common(h12);
};

auto formulaC = [&](double h12) {
return (-1) * (1 + common(h12));
};

auto layer = [&](size_t i0, size_t j0, size_t i1, size_t j1, size_t i2, size_t j2,
double h12, const vec2d<double>& u) {
return (-1) * (u[k][i0][j0] * (1 - common(h12)) + 0.5 * common(h12) * (u[k][i1][j1] + u[k][i2][j2]) + (step[2] / 2) * F[k][i0][j0]);
};


A[k] = B[k] = (phase == 0)
? formulaAB(step[0])
: formulaAB(step[1]);

C[k] = (phase == 0)
? formulaC(step[0])
: formulaC(step[1]);

if (phase == 0) {
for (size_t i = 1; i < grid[0] - 1; i++) {
for (size_t j = 1; j < grid[1] - 1; j++) {
Phi[k][i][j] = layer(i, j, i, j - 1, i, j + 1, step[1], uLow);
}
}

} else if (phase == 1) {
for (size_t i = 1; i < grid[0] - 1; i++) {
for (size_t j = 1; j < grid[1] - 1; j++) {
Phi[k][i][j] = layer(i, j, i - 1, j, i + 1, j, step[0], uMid);
}
}

} else {
throw std::runtime_error(AppConstansts::ALARM_LAYERS_PARAMS);
}
}
}

void executeSerialSweepLayers(size_t phase) {
kappa = std::make_pair(1., 1.);
mu = std::make_pair(0., 0.);
gamma = std::make_pair(1., 1.);

if (phase == 0) {
this->executeSerialSweepLayersPhase1();

} else if (phase == 1) {
this->executeSerialSweepLayersPhase2();

} else {
throw std::runtime_error(AppConstansts::ALARM_EXECUTE_LAYERS);
}
}

void executeParallelSweepLayers(size_t phase) {
kappa = std::make_pair(1., 1.);
mu = std::make_pair(0., 0.);
gamma = std::make_pair(1., 1.);

if (phase == 0) {
this->executeParallelSweepLayersPhase1();

} else if (phase == 1) {
this->executeParallelSweepLayersPhase2();

} else {
throw std::runtime_error(AppConstansts::ALARM_EXECUTE_LAYERS);
}
}
};