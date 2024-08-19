#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "Cluster.hpp"
#include "Point.hpp"
#include "RNG.hpp"


namespace KMeans {

#pragma omp declare reduction(FloatVectorMin : std::vector<float> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::min<>()))

#pragma omp declare reduction(DoubleVectorMin : std::vector<double> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::min<>()))

#pragma omp declare reduction(FloatVectorDiv : std::vector<float> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(),  [](float& x, float& y){return x/=y; }))

#pragma omp declare reduction(DoubleVectorDiv : std::vector<double> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), [](double& x, double& y){return x/=y; }))

#pragma omp declare reduction(FloatVectorSum : std::vector<float> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(),  [](float& x, float& y){return x+=y; }))

#pragma omp declare reduction(DoubleVectorSum : std::vector<double> : \
std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), [](double& x, double& y){return x+=y; }))


void initOpenMP(int num_threads) {
if (num_threads > 0) 
omp_set_num_threads(num_threads);
}


template <typename T>
void computeDistances(T &sum, std::vector<T> &minDist, std::size_t i, std::size_t nData,
const std::vector<Point<T>> &data, Cluster<T> &C)
{
for (std::size_t j = 0; j < nData; j++) {
T dist = ParSqEuclidianDist(data[j], C);
minDist[i] = std::min(minDist[i], dist);
sum += minDist[j];
}
}


template <typename T>
auto ParInitClusters(const std::vector<Point<T>> &data, const std::size_t &nClusters)
{

auto gen         = getGenerator();
const auto nData = data.size();

std::uniform_int_distribution<std::size_t> unifIntDist(0, nData - 1);

std::size_t nCluster   = unifIntDist(gen);
auto firstClusterCoord = data[nCluster].GetCoord();
auto firstCluster      = Cluster<T>(firstClusterCoord);
const auto inf         = std::numeric_limits<T>::infinity();

std::vector<Cluster<T>> clusters({firstCluster});
std::vector<T> minDist(nData, inf);
std::vector<T> minPropDist(nData);

for (std::size_t i = 0; i < nClusters; i++) {
auto sum = T(.0);
auto &C = clusters.back();
std::vector<T> dist(nData, T(.0));
auto &coord = C.GetCoord();

if constexpr (std::is_same<T, float>::value) {
#pragma omp parallel for firstprivate(nData, data, C) schedule(static) reduction(FloatVectorMin:minDist) reduction(+:sum)
computeDistances(sum, minDist, i, nData, data, C);
} else {
#pragma omp parallel for firstprivate(nData, data, C) schedule(static) reduction(FloatVectorMin:minDist) reduction(+:sum)
computeDistances(sum, minDist, i, nData, data, C);
}

#pragma omp parallel for firstprivate(nData, minDist, sum) shared(minPropDist) schedule(static)
for (std::size_t j = 0; j < nData; j++)
#pragma omp critical
minPropDist[j] = minDist[j] / sum;

if (i < nClusters - 1) {
std::discrete_distribution<std::size_t> wUnifIntDist(minPropDist.begin(), minPropDist.end());
nCluster = wUnifIntDist(gen);
auto nextClusterCoord = data[nCluster].GetCoord();
auto nextCluster = Cluster<T>(nextClusterCoord);
clusters.push_back(nextCluster);
}
}
return clusters;
}


template <typename T>
T ParSqEuclidianDist(const Point<T> &point, const Cluster<T> &cluster)
{
auto ndim          = point.GetDim();
auto &pointCoord   = point.GetCoord();
auto &clusterCoord = cluster.GetCoord();
T dist             = T(.0);

for (std::size_t i = 0; i < ndim; i++)
dist += (pointCoord[i] - clusterCoord[i]) * (pointCoord[i] - clusterCoord[i]);
return dist;
}


template <typename T>
std::size_t ParAssignPoints(std::vector<Point<T>> &data, std::vector<Cluster<T>> &clusters)
{
std::size_t nData      = data.size();
std::size_t nClusters  = clusters.size();
std::size_t nReasigned = 0;

#pragma omp parallel for shared(data, clusters, nReasigned) firstprivate(nData, nClusters) \
schedule(static) reduction(+:nReasigned)
for (std::size_t i = 0; i < nData; i++) {
auto &P = data[i];
T minDist = ParSqEuclidianDist(P, clusters[0]);
std::size_t clusterId = 0;
for (std::size_t nCluster = 0; nCluster < nClusters; nCluster++) {
auto &C = clusters[nCluster];
T dist = ParSqEuclidianDist(P, C);
if (dist < minDist) {
minDist = dist;
clusterId = nCluster;
}
}
if (P.GetClusterId() != clusterId || !P.IsAssigned()) {
#pragma omp critical
P.SetClusterId(clusterId);
nReasigned++;
}
#pragma omp critical
clusters[clusterId].Add(i);
}


return nReasigned;
}


template <typename T>
void reduceCoordDiv(std::size_t dim, std::size_t size, std::vector<T> &coord) {
for (std::size_t i = 0; i < dim; i++)
coord[i] /= T(size);
}


template <typename T>
void ParUpdateClusters(const std::vector<Point<T>> &data, std::vector<Cluster<T>> &clusters)
{
std::size_t k   = clusters.size();
std::size_t dim = clusters[0].GetDim();

for (std::size_t nCluster = 0; nCluster < k; nCluster++) {
std::vector<T> coord(dim, T(.0));
auto size = clusters[nCluster].GetSize();
auto pointsId = clusters[nCluster].GetPointsId();

if constexpr (std::is_same<T, float>::value) {
for(std::size_t i = 0; i < dim; i++) {

#pragma omp parallel for shared(coord) firstprivate(size, data, pointsId) reduction(FloatVectorSum \
: coord) schedule(static)
for(std::size_t j = 0; j < size; j++) {
auto &pCoord = data[pointsId[j]].GetCoord();
coord[i] += pCoord[i];
}
}

#pragma omp parallel for firstprivate(dim, size) reduction(FloatVectorDiv : coord) schedule(static)
reduceCoordDiv(dim, size, coord);
} else {
for(std::size_t i = 0; i < dim; i++) {

#pragma omp parallel for shared(coord) firstprivate(size, data, pointsId) reduction(DoubleVectorSum \
: coord) schedule(static)
for(std::size_t j = 0; j < size; j++) {
auto &pCoord = data[pointsId[j]].GetCoord();
coord[i] += pCoord[i];
}
}

#pragma omp parallel for firstprivate(dim, size) reduction(DoubleVectorDiv : coord) schedule(static)
reduceCoordDiv(dim, size, coord);
}

clusters[nCluster].SetCoord(coord);
clusters[nCluster].Clear();
}
}


template <typename T>
auto ParKMeans(std::vector<Point<T>> &data, const std::size_t &nClusters,
const std::size_t &iterMax, const T threshold)
-> std::vector<Cluster<T>>
{
auto clusters          = ParInitClusters(data, nClusters);
std::size_t nData      = data.size();
std::size_t nIter      = 0;
std::size_t nReasigned = data.size();
T fReasigned           = T(1.0);

while (nIter < iterMax && fReasigned > threshold) {
nReasigned = ParAssignPoints(data, clusters);
ParUpdateClusters(data, clusters);
fReasigned = nReasigned / T(nData);
nIter++;
}
if (fReasigned > threshold)
throw std::runtime_error("Parallel KMeans algorithm did not converge.");
return clusters;
}

} 
