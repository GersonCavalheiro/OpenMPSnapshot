

#include "trinity/metric.h"

namespace trinity {

Metrics::Metrics(
Mesh*  input_mesh,
double target_factor,
int    Lp_norm,
double h_min,
double h_max
) : mesh    (input_mesh),
nb_nodes(mesh->nb.nodes),
nb_elems(mesh->nb.elems),
nb_cores(mesh->nb.cores),
verbose (mesh->param.verb),
iter    (mesh->param.iter),
rounds  (mesh->param.rounds) {

param.chunk      = mesh->nb.nodes / mesh->nb.cores;
param.target     = (int) std::floor(mesh->nb.nodes * target_factor);
param.norm       = Lp_norm;
param.h_min      = h_min;
param.h_max      = h_max;
param.scale.fact = 0;
param.scale.exp  = Lp_norm ? -1. / (2 * Lp_norm + 2) : 0.5;
param.eigen.min  = 1 / std::pow(h_min, 2);
param.eigen.max  = 1 / std::pow(h_max, 2);

field.complexity = 0;
field.solut      = mesh->geom.solut.data();
field.tensor     = mesh->geom.tensor.data();
field.stencil    = new Patch[nb_nodes];
field.gradient   = new double[nb_nodes * 2];
}


Metrics::~Metrics() {}


void Metrics::run(Stats* total) {
#pragma omp parallel
{
initialize();
recoverHessianField();
normalizeLocally();
computeComplexity();
normalizeGlobally();
recap(total);
}
clear();
}


void Metrics::clear() {

delete[] field.stencil;
delete[] field.gradient;
}


void Metrics::recoverHessianField() {

#pragma omp for
for (int i = 0; i < nb_nodes; ++i) {
field.stencil[i] = mesh->getVicinity(i, 2);
computeGradient(i);
}

#pragma omp for
for (int i = 0; i < nb_nodes; ++i)
computeHessian(i);

#pragma omp master
{
mesh->geom.solut.clear();
mesh->geom.solut.shrink_to_fit();
}
}


void Metrics::normalizeLocally() {

double m[9];
double h[4];
double val[2], vec[4];

#pragma omp for schedule(guided)
for (int i = 0; i < nb_nodes; ++i) {
int k = i * 3;
std::memset(val, 0, sizeof(double) * 2);
std::memset(vec, 0, sizeof(double) * 4);

m[0] = field.tensor[k];
m[1] = field.tensor[k + 1];
m[2] = m[1];
m[3] = field.tensor[k + 2];

numeric::eigenDecompose(m, val, vec, vec + 2);
val[0] = std::max(std::abs(val[0]), EPSILON);
val[1] = std::max(std::abs(val[1]), EPSILON);



double const det = val[0] * val[1];
double const scale_loc = std::pow(det, param.scale.exp);
val[0] *= scale_loc;
val[1] *= scale_loc;

m[0] = val[0] * vec[0];
m[1] = val[0] * vec[2];
m[2] = val[1] * vec[1];
m[3] = val[1] * vec[3];

h[0] = vec[0] * m[0] + vec[1] * m[2];
h[1] = vec[0] * m[1] + vec[1] * m[3];
h[2] = vec[2] * m[0] + vec[3] * m[2];
h[3] = vec[2] * m[1] + vec[3] * m[3];
assert(std::abs(h[1] - h[2]) < 1.e-3);

field.tensor[k]     = h[0];
field.tensor[k + 1] = h[1];
field.tensor[k + 2] = h[3];
}
}


void Metrics::computeComplexity() {

double s[4];
double p[6];
double phi = 0.;
#pragma omp for schedule(guided) nowait
for (int i = 0; i < nb_elems; ++i) {
auto v = mesh->getElemCoord(i, p);
s[0] = p[2] - p[0];      
s[1] = p[4] - p[0];      
s[2] = p[3] - p[1];      
s[3] = p[5] - p[1];      
double const area = 0.5 * (s[0] * s[3] - s[1] * s[2]);

double rho = 0.;
for (int j = 0; j < 3; ++j) {
int k = 3 * v[j]; 
double const det =
field.tensor[k] * field.tensor[k + 2] - pow(field.tensor[k + 1], 2);
assert(det > 0);
rho += sqrt(det);
}
rho /= 3;
phi += rho * area;
}
#pragma omp critical
field.complexity += phi;
#pragma omp barrier
}


void Metrics::normalizeGlobally() {

assert(field.complexity);

int    j;
double m[9];
double h[4];
double val[2], vec[4];

#pragma omp single
param.scale.fact = param.target / field.complexity;

#pragma omp for schedule(guided)
for (int i = 0; i < nb_nodes; ++i) {

int k = i * 3;
m[0] = field.tensor[k];
m[1] = field.tensor[k + 1];
m[2] = m[1];
m[3] = field.tensor[k + 2];

numeric::eigenDecompose(m, val, vec, vec + 2);
val[0] *= param.scale.fact;
val[1] *= param.scale.fact;
for (j = 0; j < 2; ++j) {
val[j] = std::min(val[j], param.eigen.min);
val[j] = std::max(val[j], param.eigen.max);
}



m[0] = val[0] * vec[0];
m[1] = val[0] * vec[2];
m[2] = val[1] * vec[1];
m[3] = val[1] * vec[3];

h[0] = vec[0] * m[0] + vec[1] * m[2];
h[1] = vec[0] * m[1] + vec[1] * m[3];
h[2] = vec[2] * m[0] + vec[3] * m[2];
h[3] = vec[2] * m[1] + vec[3] * m[3];
assert(std::abs(h[1] - h[2]) < 1.e-3);

field.tensor[k]     = h[0];
field.tensor[k + 1] = h[1];
field.tensor[k + 2] = h[3];
}
}



void Metrics::initialize() {
#pragma omp master
{
assert(param.norm < 5);
assert(param.h_min > 0);
assert(param.h_max > 0);

if (not verbose) {
std::printf("\n\r= Remeshing  ... %3d %% =", 0);
} else if (verbose == 1) {
std::printf("%-18s%s", "= metric field", "...");
} else if (verbose == 2) {
std::printf("Computing multi-scale metric field ... \n");
}

std::fflush(stdout);
time.start = timer::now();
}

int rank = omp_get_thread_num();
int off  = rank * param.chunk;

std::memset(field.gradient + (off * 2), 0, (param.chunk * 2) * sizeof(double));
std::memset(  field.tensor + (off * 3), 0, (param.chunk * 3) * sizeof(double));
}


void Metrics::recap(Stats* total) {
#pragma omp master
{
int end = timer::elapsed_ms(time.start);

if (total != nullptr) {
total->eval += nb_nodes;
total->task += nb_nodes;
total->elap += end;
}

if (not verbose) {
std::printf(
"\r= Remeshing  ... %3d %% =",
(int) std::floor(100 * (++iter) / (4 * rounds + 1))
);
} else if (verbose == 1) {
std::printf(
"%10d task/sec \e[32m(%4.2f s)\e[0m\n",
(int) std::floor(nb_nodes / (end * 1e-3)), (float) end / 1e3
);
} else if (verbose == 2) {
std::printf(
"= norm L^%s\n",
param.norm <= 0 ? "inf" : std::to_string(param.norm).data()
);
std::printf("= target : %.2e\n", (float) param.target);
std::printf("= complex: %.1f\n", (float) std::floor(field.complexity));
std::printf("= scale_f: %.1f\n", param.scale.fact);
std::printf("done. \e[32m(%d ms)\e[0m\n\n", end);
}
std::fflush(stdout);
}
}

} 
