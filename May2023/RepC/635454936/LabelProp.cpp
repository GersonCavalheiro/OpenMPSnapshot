#include <vector>
#include <set>
#include <omp.h>
#include "../include/GraphComponents.h"
std::vector<unsigned int> generateLabels(int length) {
std::vector<unsigned int> ret;
ret.resize(length);
for (int i = 0; i < length; i++) {
ret[i] = i;
}
return ret;
}
std::vector<Vertex*> getConnectedComponents(Graph& graph, const unsigned int threads) {
std::vector<Vertex*> connectedComponentsList;
std::vector<unsigned int> labels = generateLabels(graph.vertices.size());
bool change;
do {
change = false;
#pragma omp parallel for num_threads(threads)
for (int i = 0; i < graph.vertices.size(); i++) {
Vertex v = *graph.vertices[i];
std::vector<unsigned int> incidentEdges = v.incidentEdges;
unsigned int min = labels[v.id];
for (int edge = 0; edge < incidentEdges.size(); edge++) {
int neighborID = incidentEdges[edge];
if (labels[neighborID] < min)
min = labels[neighborID];
}
if (labels[v.id] != min) {
labels[v.id] = min;
change = true;
}
for (int edge = 0; edge < incidentEdges.size(); edge++) {
int neighborID = incidentEdges[edge];
if (labels[neighborID] > min) {
labels[neighborID] = min;
change = true;
}
}
}
} while (change);
std::set<unsigned int> ccLabels;
for (int i = 0; i < labels.size(); i++) {
if (!ccLabels.contains(labels[i])) {
ccLabels.insert(labels[i]);
connectedComponentsList.push_back(graph.vertices[i]);
}
}
return connectedComponentsList;
}