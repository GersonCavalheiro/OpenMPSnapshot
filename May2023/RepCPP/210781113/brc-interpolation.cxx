#include "algorithm"
#include "iostream"

#include "ANN/ANN.h"

#include "constants.hpp"
#include "parameters.hpp"

#include "barycentric-fn.hpp"
#include "utils.hpp"
#include "brc-interpolation.hpp"

namespace { 

typedef Array2D<double,NODES_PER_ELEM> brc_t;


void interpolate_field(const brc_t &brc, const int_vec &el, const conn_t &connectivity,
const double_vec &source, double_vec &target)
{
#pragma omp parallel for default(none)          \
shared(brc, el, connectivity, source, target)
for (std::size_t i=0; i<target.size(); i++) {
int e = el[i];
const int *conn = connectivity[e];
double result = 0;
for (int j=0; j<NODES_PER_ELEM; j++) {
result += source[conn[j]] * brc[i][j];
}
target[i] = result;
}
}


void interpolate_field(const brc_t &brc, const int_vec &el, const conn_t &connectivity,
const array_t &source, array_t &target)
{
#pragma omp parallel for default(none)          \
shared(brc, el, connectivity, source, target)
for (std::size_t i=0; i<target.size(); i++) {
int e = el[i];
const int *conn = connectivity[e];
for (int d=0; d<NDIMS; d++) {
double result = 0;
for (int j=0; j<NODES_PER_ELEM; j++) {
result += source[conn[j]][d] * brc[i][j];
}
target[i][d] = result;
}
}
}


void prepare_interpolation(const Variables &var,
const Barycentric_transformation &bary,
const array_t &old_coord,
const conn_t &old_connectivity,
const std::vector<int_vec> &old_support,
brc_t &brc, int_vec &el)
{

double **points = new double*[old_coord.size()];
for (std::size_t i=0; i<old_coord.size(); i++) {
points[i] = const_cast<double*>(old_coord[i]);
}
ANNkd_tree kdtree(points, old_coord.size(), NDIMS);

const int k = 1;
const double eps = 0;
int nn_idx[k];
double dd[k];

for (int i=0; i<var.nnode; i++) {
double *q = (*var.coord)[i];

kdtree.annkSearch(q, k, nn_idx, dd, eps);
int nn = nn_idx[0];

const int_vec &nn_elem = old_support[nn];


double r[NDIMS];
int e;

if (dd[0] == 0) {
e = nn_elem[0];
bary.transform(q, e, r);
for (int d=0; d<NDIMS; d++) {
if (r[d] > 0.9)
r[d] = 1;
else
r[d] = 0;
}
goto found;
}

for (std::size_t j=0; j<nn_elem.size(); j++) {
e = nn_elem[j];
bary.transform(q, e, r);
if (bary.is_inside(r)) {
goto found;
}
}



{


int_vec searched(nn_elem);

for (std::size_t j=0; j<nn_elem.size(); j++) {
int ee = nn_elem[j];
const int *conn = old_connectivity[ee];
for (int m=0; m<NODES_PER_ELEM; m++) {
int np = conn[m];
const int_vec &np_elem = old_support[np];
for (std::size_t j=0; j<np_elem.size(); j++) {
e = np_elem[j];
auto it = std::find(searched.begin(), searched.end(), e);
if (it != searched.end()) {
continue;
}
searched.push_back(e);
bary.transform(q, e, r);
if (bary.is_inside(r)) {
goto found;
}
}
}
}
}
{

e = nn_elem[0];
bary.transform(points[nn], e, r);
}
found:
el[i] = e;
double sum = 0;
for (int d=0; d<NDIMS; d++) {
brc[i][d] = r[d];
sum += r[d];
}
brc[i][NODES_PER_ELEM-1] = 1 - sum;
}

delete [] points;

}

} 


void barycentric_node_interpolation(Variables &var,
const Barycentric_transformation &bary,
const array_t &old_coord,
const conn_t &old_connectivity)
{
int_vec el(var.nnode);
brc_t brc(var.nnode);
prepare_interpolation(var, bary, old_coord, old_connectivity, *var.support, brc, el);

double_vec *a;
a = new double_vec(var.nnode);
interpolate_field(brc, el, old_connectivity, *var.temperature, *a);
delete var.temperature;
var.temperature = a;

array_t *b = new array_t(var.nnode);
interpolate_field(brc, el, old_connectivity, *var.vel, *b);
delete var.vel;
var.vel = b;

b = new array_t(var.nnode);
interpolate_field(brc, el, old_connectivity, *var.coord0, *b);
delete var.coord0;
var.coord0 = b;

}


void barycentric_node_interpolation_forT(const Variables &var,
const Barycentric_transformation &bary,
const array_t &input_coord,
const conn_t &input_connectivity,
const std::vector<int_vec> &input_support,
const double_vec &inputtemperature,
double_vec &outputtemperature)
{
int_vec el(var.nnode);
brc_t brc(var.nnode);
prepare_interpolation(var, bary, input_coord, input_connectivity, input_support, brc, el);

interpolate_field(brc, el, input_connectivity, inputtemperature, outputtemperature);
}
