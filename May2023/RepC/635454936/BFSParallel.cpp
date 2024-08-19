#include <vector>
#include <omp.h>
#include "../include/GraphComponents.h"
void parallelBFS(const std::vector<Vertex*>& vertices, const unsigned int src, std::vector<int>& levels, const unsigned int threads) {
levels[src] = 0;
int currentLevel = 0;
std::vector<unsigned int> frontier;
frontier.push_back(src);
unsigned int offset = 0;
#pragma omp parallel num_threads(threads)
while (!frontier.empty()) {
std::vector<unsigned int> localNextFrontier;
localNextFrontier.reserve(frontier.size());
#pragma omp for schedule(dynamic)
for (int i = 0; i < frontier.size(); i++) {
unsigned int frontierVertex = frontier[i];
std::vector<unsigned int> incidentEdges = vertices[frontierVertex]->incidentEdges;
for (unsigned int toIdx : incidentEdges) {
if (levels[toIdx] == -1) {
int insert;
#pragma omp atomic capture
{
insert = levels[toIdx];
levels[toIdx] = currentLevel + 1;
}
if (insert == -1) {
localNextFrontier.push_back(toIdx);
}
}
}
}
unsigned int localOffset;
#pragma omp atomic capture
{
localOffset = offset;
offset += localNextFrontier.size();
}
#pragma omp barrier
#pragma omp single
{
frontier.resize(offset);
offset = 0;
currentLevel++;
}
#pragma omp barrier
for (int i = 0; i < localNextFrontier.size(); i++) {
frontier[i + localOffset] = localNextFrontier[i];
}
#pragma omp barrier
}
}
std::vector<Vertex*> getConnectedComponents(Graph& graph, const unsigned int threads) {
std::vector<Vertex*> connectedComponentsList;
std::vector<int> levels(graph.vertices.size(), -1);
for (int i = 0; i < graph.vertices.size(); i++) {
if (levels[i] == -1) {
parallelBFS(graph.vertices, i, levels, threads);
connectedComponentsList.push_back(graph.vertices[i]);
}
}
return connectedComponentsList;
}
