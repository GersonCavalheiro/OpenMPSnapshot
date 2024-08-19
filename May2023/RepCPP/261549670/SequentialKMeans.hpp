#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

#include "Cluster.hpp"
#include "Point.hpp"
#include "RNG.hpp"


namespace KMeans {

template <typename T>
auto SeqInitClusters(const std::vector<Point<T>> &data, const std::size_t &nClusters)
{
auto gen   = getGenerator();
auto nData = data.size();

std::uniform_int_distribution<std::size_t> unifIntDist(0, nData - 1);

std::size_t nCluster   = unifIntDist(gen);
auto firstClusterCoord = data[nCluster].GetCoord();
auto firstCluster      = Cluster<T>(firstClusterCoord);

std::vector<Cluster<T>> clusters({firstCluster});

auto inf = std::numeric_limits<T>::infinity();

std::vector<T> minDist(nData, inf);
std::vector<T> minPropDist(nData);

for (std::size_t i = 0; i < nClusters; i++) {
auto sum = T(.0);
for (std::size_t j = 0; j < nData; j++) {
T dist = SeqSqEuclidianDist(data[j], clusters.back());
if (dist < minDist[j])
minDist[j] = dist;
sum += minDist[j];
}
for (std::size_t j = 0; j < nData; j++)
minPropDist[j] = minDist[j] / sum;
if (i < nClusters - 1) {
std::discrete_distribution<std::size_t> wUnifIntDist(minPropDist.begin(),
minPropDist.end());
nCluster = wUnifIntDist(gen);
auto nextClusterCoord = data[nCluster].GetCoord();
auto nextCluster = Cluster<T>(nextClusterCoord);
clusters.push_back(nextCluster);
}
}
return clusters;
}


template <typename T>
T SeqSqEuclidianDist(const Point<T> &point, const Cluster<T> &cluster)
{
auto ndim          = point.GetDim();
auto &pointCoord   = point.GetCoord();
auto &clusterCoord = cluster.GetCoord();
T dist             = T(.0);

for (std::size_t i = 0; i < ndim; i++) {
auto d = std::pow(pointCoord[i] - clusterCoord[i], 2);
dist += d;
}
return dist;
}


template <typename T>
std::size_t SeqAssignPoints(std::vector<Point<T>> &data, std::vector<Cluster<T>> &clusters)
{
std::size_t nData      = data.size();
std::size_t nClusters  = clusters.size();
std::size_t nReasigned = 0;

for (std::size_t i = 0; i < nData; i++) {
auto &P = data[i];
T minDist = SeqSqEuclidianDist(P, clusters[0]);
std::size_t clusterId = 0;
for (std::size_t nCluster = 0; nCluster < nClusters; nCluster++) {
auto &C = clusters[nCluster];
T dist = SeqSqEuclidianDist(P, C);
if (dist < minDist) {
minDist = dist;
clusterId = nCluster;
}
}
if (P.GetClusterId() != clusterId || !P.IsAssigned()) {
P.SetClusterId(clusterId);
nReasigned++;
}
clusters[clusterId].Add(i);
}
return nReasigned;
}


template <typename T>
void SeqUpdateClusters(const std::vector<Point<T>> &data, std::vector<Cluster<T>> &clusters)
{
for (auto &C : clusters) {
C.Update(data);
C.Clear();
}
}


template <typename T>
auto SeqKMeans(std::vector<Point<T>> &data, const std::size_t &nClusters, 
const std::size_t &iterMax, const T threshold)
-> std::vector<Cluster<T>>
{
auto clusters          = SeqInitClusters(data, nClusters);
std::size_t nData      = data.size();
std::size_t nIter      = 0;
std::size_t nReasigned = data.size();
T fReasigned           = T(1.0);

while (nIter < iterMax && fReasigned > threshold) {
nReasigned = SeqAssignPoints(data, clusters);
SeqUpdateClusters(data, clusters);
fReasigned = nReasigned / T(nData);
nIter++;
}
if (fReasigned > threshold)
throw std::runtime_error("Sequential KMeans algorithm did not converge.");
return clusters;
}

} 
