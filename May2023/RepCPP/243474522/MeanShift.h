#pragma once

#include <utility>
#include <vector>
#include <limits>
#include <iterator>
#include <type_traits>
#include <stdexcept>
#include <cmath>

#ifdef _OPENMP

#include <omp.h>
#include <chrono>
#include "Cluster.h"
#include "Point.h"

#endif

class MeanShift {
private:
std::vector<Point *> v;
std::vector<Point *> shifted;
double kcost;
double bandwidth;
double bw_sq;

public:
MeanShift(std::vector<Point *> w, double bandwidth) {
this->bandwidth = bandwidth;
bw_sq = bandwidth * bandwidth;
kcost = 1.0 / (bandwidth * std::sqrt(2 * M_PI));
v = std::move(w); 
shifted.resize(v.size());
for (auto i = 0; i < v.size(); i++) {
shifted[i] = new Point();
for (int k = 0; k < 3; k++) {
shifted[i]->setCoord(k, v[i]->getCoord(k));
}
}
}

Point *mean_shift(Point *point, int dim) {
Point *shift = new Point(dim);
double total_weight = 0.0;
for (auto it = 0; it < v.size(); it++) {
auto dist = distance(v[it], point);
double weight = kernel(dist);
for (int k = 0; k < dim; k++) {
double c = shift->getCoord(k);
shift->setCoord(k, c + ((v[it]->getCoord(k)) * weight));
}

total_weight += weight;
}
for (int k = 0; k < 3; k++) {
double real = shift->getCoord(k);
shift->setCoord(k, real / (total_weight));
}

return shift;
}


std::vector<Point *> shifter_seq(double epsilon = 100.0, int max_iter = 100) {

for (auto i = 0; i < shifted.size(); i++) {
Point *pt;
Point *point;
int iter = 0;
double d = 0;
do {
pt = shifted[i];
point = mean_shift(pt, 3);
d = distance(pt, point);
shifted[i] = point;
iter++;
} while (d > epsilon && iter < max_iter);
}


return shifted;
}

std::vector<Point *> shifter(double epsilon = 5000.0,
int max_iter = 100, int N = omp_get_max_threads(), bool doStatic = false) {

omp_set_num_threads(N);
#pragma omp parallel for 
for (auto i = 0; i < shifted.size(); i++) {
Point *pt;
Point *point;
int iter = 0;
double d = 0;
do {
pt = shifted[i];
point = mean_shift(pt, 3);
d = distance(pt, point);
shifted[i] = point;
iter++;
} while (d > epsilon && iter < max_iter);
}
return shifted;
}


std::vector<Point *> shifter_2(double epsilon = 5000.0,
int max_iter = 100, int N = omp_get_max_threads(), bool doStatic = false) {

omp_set_num_threads(N);
#pragma omp parallel for schedule(dynamic)
for (auto i = 0; i < shifted.size(); i++) {
Point *pt;
Point *point;
int iter = 0;
double d = 0;
do {
pt = shifted[i];
point = mean_shift(pt, 3);
d = distance(pt, point);
shifted[i] = point;
iter++;
} while (d > epsilon && iter < max_iter);
}

return shifted;
}

std::vector<Cluster *> cluster_shifted(std::vector<Point *> shi, int dim, double epsilon = 5000.0) {
if (dim <= 0)
throw std::invalid_argument("Dimension must be greater than 0");
std::vector<Cluster *> clusters;
for (int it = 0; it < shi.size(); it++) {
Point *pt = shi[it];
std::size_t c = 0;
for (; c < clusters.size(); c++) {
if (distance(pt, clusters[c]->getMode()) <= epsilon) {
break;
}
}
if (c == clusters.size()) {
Cluster *pCluster = new Cluster(pt);
clusters.emplace_back(pCluster);
}
clusters[c]->members.emplace_back(it);


}

return clusters;
}

std::vector<Cluster *>
mean_shift_cluster(double epsilon = 5000.0, int max_iter = 100, int N = omp_get_max_threads(),
bool doStatic = false) {
if (doStatic) {
auto shi = shifter(epsilon, max_iter, N, doStatic);
return cluster_shifted(shi, 3, epsilon);
} else {
auto shi = shifter_2(epsilon, max_iter, N, doStatic);
return cluster_shifted(shi, 3, epsilon);
}
}


std::vector<Cluster *> mean_shift_cluster_seq(double epsilon = 100.0,
int max_iter = 100) {

auto shi = shifter_seq(epsilon, max_iter);
return cluster_shifted(shi, 3, epsilon);

}


static double distance(Point *v1, Point *v2) {
double dist = (v1->getCoord(0) - v2->getCoord(0)) * (v1->getCoord(0) - v2->getCoord(0)) +
(v1->getCoord(1) - v2->getCoord(1)) * (v1->getCoord(1) - v2->getCoord(1)) +
(v1->getCoord(2) - v2->getCoord(2)) * (v1->getCoord(2) - v2->getCoord(2));
return dist;
}


double kernel(double d) {
return (kcost * std::exp(d / (-2.0 * (bw_sq))));
}


};

