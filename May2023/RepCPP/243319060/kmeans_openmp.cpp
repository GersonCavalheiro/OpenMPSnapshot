#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include "kmeans_openmp.h"

double KmeansOpenMP::costFunction(const std::vector<Observation> &points, const std::vector<Observation> &centroids)
{
double totalDistance = 0;
#pragma omp parallel for reduction(+ \
: totalDistance)
for (const auto &p : points)
for (const auto &c : centroids)
totalDistance += Point::distance(p, c);
return totalDistance;
}

void KmeansOpenMP::assignToClosestCentroid(Observation &point, const std::vector<Observation> &centroids)
{
double distance = std::numeric_limits<double>::max();
for (const auto &centroid : centroids)
{
const auto newDistance = Point::distance(point.getPoint(), centroid.getPoint());
if (newDistance < distance)
{
point.setClusterId(centroid.getClusterId());
distance = newDistance;
}
}
}

void KmeansOpenMP::updateCentroids(const std::vector<Observation> &initPoints, std::vector<Observation> &centroids)
{
#pragma omp parallel for
for (auto &centroid : centroids)
{
double x = 0, y = 0;
int counter = 0;
#pragma omp parallel for default(shared) reduction(+ \
: x, y, counter)
for (const auto &p : initPoints)
{
if (p.getClusterId() == centroid.getClusterId())
{
x += p.getX();
y += p.getY();
++counter;
}
}
if (counter > 0)
{
x /= counter;
y /= counter;
centroid.setX(x);
centroid.setY(y);
}
}
}

ObservationsWithIterations KmeansOpenMP::fit(std::vector<Observation> &initPoints, unsigned int k, double tolerance, int maxIteration)
{
std::vector<Observation> centroids;
std::vector<int> positions(initPoints.size());
std::iota(positions.begin(), positions.end(), 0);
std::random_shuffle(positions.begin(), positions.end(), [](const int i) { return rand() % i; });
for (unsigned int i = 0; i < k; ++i)
{
const auto point = initPoints.at(positions.at(i)).getPoint();
centroids.emplace_back(Observation(point.getX(), point.getY(), i));
}

for (int i = 0; i < maxIteration; ++i)
{
double costOld = costFunction(initPoints, centroids);
#pragma omp parallel for
for (auto &point : initPoints)
assignToClosestCentroid(point, centroids);

updateCentroids(initPoints, centroids);
double costNew = costFunction(initPoints, centroids);
if (std::abs(costOld - costNew) < tolerance)
{
return std::make_pair(centroids, i);
}
}
return std::make_pair(centroids, maxIteration);
}