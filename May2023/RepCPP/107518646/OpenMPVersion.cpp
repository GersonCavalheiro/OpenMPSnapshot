#include "../include/OpenMPVersion.h"

OpenMPVersion::OpenMPVersion(std::string graphFilename, unsigned vertexesNumber){
this->init(graphFilename, vertexesNumber);
}


AbstractGraph::path * OpenMPVersion::getCriticalPath(unsigned vertexStart) {

#pragma omp parallel for

for (int i = 0; i < vertexesNumber; i++)	
for (int j = 0; j < vertexesNumber; j++)
matrix[i][j] = -matrix[i][j];

path* res = new path();

std::pair<std::vector<long>, std::vector<unsigned>> pair;
bellmanFord(vertexStart, &pair);
int intIndex = std::min_element(pair.first.begin(), pair.first.end()) - pair.first.begin();
res->pathLength = -pair.first[intIndex];


return res;
}

AbstractGraph::path * OpenMPVersion::getCriticalPath() {
return getCriticalPath(0);
}

void OpenMPVersion::bellmanFord(unsigned row, std::pair<std::vector<long>, std::vector<unsigned>>* pair) {
std::vector<long> distance(vertexesNumber);
std::vector<unsigned> predecessor;

#pragma omp parallel for

for (int i = 0; i < vertexesNumber; i++) {
distance[i] = LONG_MAX;
}

distance[row] = 0;

#pragma omp parallel

for (int i = 0; i < vertexesNumber; i++) {
for (int j = 0; j < vertexesNumber; j++) {
if (matrix[i][j] != 0) {
if (distance[j] > distance[i] + matrix[i][j]) {
distance[j] = distance[i] + matrix[i][j];
}
}
}
}

#pragma omp barrier
#pragma omp single

pair->first = distance;
pair->second = predecessor;
}
