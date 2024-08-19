#include "iostream"
#include <map>
#include <numeric>
#include <stdexcept>

#include "ANN/ANN.h"

#include "constants.hpp"
#include "parameters.hpp"
#include "barycentric-fn.hpp"
#include "mesh.hpp"
#include "utils.hpp"
#include "nn-interpolation.hpp"


namespace {

void find_nearest_neighbor(const Variables &var, ANNkd_tree &kdtree,
int_vec &idx, int_vec &is_changed)
{
double **new_center = elem_center(*var.coord, *var.connectivity);

const int k = 1;
const double eps = 1e-15;
int nn_idx[k];
double dd[k];
for(int e=0; e<var.nelem; e++) {
double *q = new_center[e];
kdtree.annkSearch(q, k, nn_idx, dd, eps);
idx[e] = nn_idx[0];
is_changed[e] = (dd[0] < eps)? 0 : 1;
}

delete [] new_center[0];
delete [] new_center;
}


void find_acm_elem_ratios(const Variables &var,
const Barycentric_transformation &bary,
int_vec &is_changed,
ANNkd_tree &kdtree,
std::vector<int_vec> &elems_vec,
std::vector<double_vec> &ratios_vec)
{
const int neta0 = 10; 
const int neta1 = neta0 + 1; 
const int neta2 = neta0;
const double spacing0 = 1.0 / neta0;
const double spacing1 = 1.0 / neta1;
const double spacing2 = 1.0 / neta2;
const int max_el = std::min(32, kdtree.nPoints());
const double eps = 0;
int nn_idx[32];
double dd[32];

const int nelem_changed = std::accumulate(is_changed.begin(), is_changed.end(), 0);
elems_vec.reserve(nelem_changed);
ratios_vec.reserve(nelem_changed);

std::map<int, int> elem_count;

for(int e=0; e<var.nelem; e++) {
if (is_changed[e]) {




const int* conn = (*var.connectivity)[e];
for (int i=0; i<neta0; i++)
for (int j=0; j<neta1; j++) {
#ifdef THREED
for (int k=0; k<neta2; k++) {
double eta[4] = {(i + 0.5) * spacing0,
(j + 0.5) * spacing1,
(k + 0.5) * spacing2,
1 - (i + 0.5) * spacing0 - (j + 0.5) * spacing1 - (k + 0.5) * spacing2};
#else
double eta[3] = {(i + 0.5) * spacing0,
(j + 0.5) * spacing1,
1 - (i + 0.5) * spacing0 - (j + 0.5) * spacing1};
#endif
if (eta[NODES_PER_ELEM-1] < 0) continue;

double x[NDIMS] = {0}; 
for (int d=0; d<NDIMS; d++)
for (int n=0; n<NODES_PER_ELEM; n++) {
x[d] += (*var.coord)[ conn[n] ][d] * eta[n];
}

kdtree.annkSearch(x, max_el, nn_idx, dd, eps);


double r[NDIMS];
int old_e;
for (int jj=0; jj<max_el; jj++) {
old_e = nn_idx[jj];
bary.transform(x, old_e, r);
if (bary.is_inside(r)) {
goto found;
}
}


continue;

found:
try {
++ elem_count.at(old_e);
}
catch (std::out_of_range const &exc) {
elem_count[old_e] = 1;
}
#ifdef THREED
}
#endif
}

int total_count = 0;
for (auto i=elem_count.begin(); i!=elem_count.end(); ++i)
total_count += i->second;


if (total_count == 0) {
is_changed[e] = 0;
continue;
}

if (elem_count.size() == 1) {
is_changed[e] = 0;
elem_count.clear();
continue;
}

elems_vec.push_back(int_vec());
int_vec &elems = elems_vec.back();
elems.reserve(elem_count.size());

ratios_vec.push_back(double_vec());
double_vec &ratios = ratios_vec.back();
ratios.reserve(elem_count.size());

const double inv = 1.0 / total_count;
for (auto i=elem_count.begin(); i!=elem_count.end(); ++i) {
elems.push_back(i->first);
ratios.push_back(i->second * inv);
}

elem_count.clear();
}
}
}


void prepare_interpolation(Variables &var,
const Barycentric_transformation &bary,
const array_t &old_coord,
const conn_t &old_connectivity,
int_vec &idx,
int_vec &is_changed,
std::vector<int_vec> &elems_vec,
std::vector<double_vec> &ratios_vec)
{
double **old_center = elem_center(old_coord, old_connectivity);
ANNkd_tree kdtree(old_center, old_connectivity.size(), NDIMS);

find_nearest_neighbor(var, kdtree, idx, is_changed);

find_acm_elem_ratios(var, bary, is_changed, kdtree, elems_vec, ratios_vec);

delete [] old_center[0];
delete [] old_center;
}


void inject_field(const int_vec &idx,
const int_vec &is_changed,
const std::vector<int_vec> &elems_vec,
const std::vector<double_vec> &ratios_vec,
const double_vec &source,
double_vec &target)
{
#pragma omp parallel for default(none)          \
shared(idx, source, target)
for (std::size_t i=0; i<target.size(); i++) {
int n = idx[i];
target[i] = source[n];
}

int n = 0;
for (std::size_t i=0; i<target.size(); i++) {
if (is_changed[i]) {
const int_vec &elems = elems_vec[n];
const double_vec &ratios = ratios_vec[n];

target[i] = 0;
for (std::size_t j=0; j<elems.size(); j++) {
target[i] += ratios[j] * source[ elems[j] ];
}
n ++;
}
}
}


void inject_field(const int_vec &idx,
const int_vec &is_changed,
const std::vector<int_vec> &elems_vec,
const std::vector<double_vec> &ratios_vec,
const tensor_t &source,
tensor_t &target)
{
#pragma omp parallel for default(none)          \
shared(idx, source, target)
for (std::size_t i=0; i<target.size(); i++) {
int n = idx[i];
for (int d=0; d<NSTR; d++) {
target[i][d] = source[n][d];
}
}

int n = 0;
for (std::size_t i=0; i<target.size(); i++) {
if (is_changed[i]) {
const int_vec &elems = elems_vec[n];
const double_vec &ratios = ratios_vec[n];

for (int d=0; d<NSTR; d++) {
target[i][d] = 0;
for (std::size_t j=0; j<elems.size(); j++) {
target[i][d] += ratios[j] * source[ elems[j] ][d];
}
}
n ++;
}
}
}


void nn_interpolate_elem_fields(Variables &var,
const int_vec &idx,
const int_vec &is_changed,
const std::vector<int_vec> &elems_vec,
const std::vector<double_vec> &ratios_vec)
{
const int n = var.nnode;
const int e = var.nelem;

double_vec *a;

a = new double_vec(e);
inject_field(idx, is_changed, elems_vec, ratios_vec, *var.plstrain, *a);
delete var.plstrain;
var.plstrain = a;

a = new double_vec(e);
inject_field(idx, is_changed, elems_vec, ratios_vec, *var.delta_plstrain, *a);
delete var.delta_plstrain;
var.delta_plstrain = a;

tensor_t *b;
b = new tensor_t(e);
inject_field(idx, is_changed, elems_vec, ratios_vec, *var.strain, *b);
delete var.strain;
var.strain = b;

b = new tensor_t(e);
inject_field(idx, is_changed, elems_vec, ratios_vec, *var.stress, *b);
delete var.stress;
var.stress = b;

a = new double_vec(e);
inject_field(idx, is_changed, elems_vec, ratios_vec, *var.stressyy, *a);
delete var.stressyy;
var.stressyy = a;
}

} 


void nearest_neighbor_interpolation(Variables &var,
const Barycentric_transformation &bary,
const array_t &old_coord,
const conn_t &old_connectivity)
{
int_vec idx(var.nelem); 
int_vec is_changed(var.nelem); 

std::vector<int_vec> elems_vec;
std::vector<double_vec> ratios_vec;

prepare_interpolation(var, bary, old_coord, old_connectivity, idx, is_changed, elems_vec, ratios_vec);


nn_interpolate_elem_fields(var, idx, is_changed, elems_vec, ratios_vec);
}
