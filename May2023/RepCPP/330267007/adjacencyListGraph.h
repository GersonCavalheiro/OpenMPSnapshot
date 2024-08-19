

#pragma once

#include "edge.h"
#include "graph.h"
#include "stlHashTable.h"

template <typename V, typename E, typename W>
class AdjacencyListGraph : public Graph<V, E, W> {
public:
AdjacencyListGraph();
virtual ~AdjacencyListGraph();

virtual vector<V> getVertices();
virtual void insertVertex(V v);
virtual void removeVertex(V v);
virtual bool containsVertex(V v);
virtual void insertEdge(V src, V dest, E label, W weight);
virtual void removeEdge(V src, V dest);
virtual bool containsEdge(V source, V destination);
virtual Edge<V, E, W> getEdge(V source, V destination);
virtual vector<Edge<V, E, W>> getEdges();
virtual vector<Edge<V, E, W>> getOutgoingEdges(V source);
virtual vector<Edge<V, E, W>> getIncomingEdges(V destination);
virtual vector<V> getNeighbors(V source);

private:
STLHashTable<V, bool> vertices;
STLHashTable<V, STLHashTable<V, pair<E, W>> *> edges;
};

template <typename V, typename E, typename W>
AdjacencyListGraph<V, E, W>::AdjacencyListGraph() {}

template <typename V, typename E, typename W>
AdjacencyListGraph<V, E, W>::~AdjacencyListGraph() {
vector<V> sources = edges.getKeys();
for (unsigned int i = 0; i < sources.size(); i++) {
delete edges.get(sources[i]);
}
}

template <typename V, typename E, typename W>
vector<V> AdjacencyListGraph<V, E, W>::getVertices() {
return vertices.getKeys();
}

template <typename V, typename E, typename W>
void AdjacencyListGraph<V, E, W>::insertVertex(V v) {
if (!vertices.contains(v)) {
vertices.insert(v, true);
}
}

template <typename V, typename E, typename W>
void AdjacencyListGraph<V, E, W>::removeVertex(V v) {
if (vertices.contains(v)) {
vertices.remove(v);
}
}

template <typename V, typename E, typename W>
bool AdjacencyListGraph<V, E, W>::containsVertex(V v) {
return vertices.contains(v);
}

template <typename V, typename E, typename W>
void AdjacencyListGraph<V, E, W>::insertEdge(V src, V dest, E label, W weight) {
if (!containsVertex(src)) {
throw runtime_error("Vertex not present in graph");
}
if (!containsVertex(dest)) {
throw runtime_error("Vertex not present in graph");
}
if (!edges.contains(src)) {
edges.insert(src, new STLHashTable<V, pair<E, W>>());
}
if (edges.get(src)->contains(dest)) {
} else {
edges.get(src)->insert(dest, pair<E, W>(label, weight));
}
}

template <typename V, typename E, typename W>
void AdjacencyListGraph<V, E, W>::removeEdge(V src, V dest) {
if (edges.contains(src) && edges.get(src)->contains(dest)) {
edges.get(src)->remove(dest);
} else {
throw runtime_error("Edge does not exist");
}
}

template <typename V, typename E, typename W>
bool AdjacencyListGraph<V, E, W>::containsEdge(V source, V destination) {
return edges.contains(source) && edges.get(source)->contains(destination);
}

template <typename V, typename E, typename W>
Edge<V, E, W> AdjacencyListGraph<V, E, W>::getEdge(V source, V destination) {
if (edges.contains(source) && edges.get(source)->contains(destination)) {
pair<E, W> data = edges.get(source)->get(destination);
return Edge<V, E, W>(source, data.first, data.second, destination);
} else {
throw runtime_error("Edge does not exist");
}
}

template <typename V, typename E, typename W>
vector<Edge<V, E, W>> AdjacencyListGraph<V, E, W>::getEdges() {
vector<Edge<V, E, W>> results;
vector<V> sources = edges.getKeys();
for (unsigned int i = 0; i < sources.size(); i++) {
vector<V> targets = edges.get(sources[i])->getKeys();
for (unsigned int j = 0; j < targets.size(); j++) {
pair<E, W> data = edges.get(sources[i])->get(targets[j]);
results.push_back(
Edge<V, E, W>(sources[i], data.first, data.second, targets[j]));
}
}
return results;
}

template <typename V, typename E, typename W>
vector<Edge<V, E, W>> AdjacencyListGraph<V, E, W>::getOutgoingEdges(V source) {
vector<Edge<V, E, W>> results;
if(!edges.contains(source)) {
return results;
}
vector<V> targets = edges.get(source)->getKeys();
for (unsigned int j = 0; j < targets.size(); j++) {
pair<E, W> data = edges.get(source)->get(targets[j]);
results.push_back(
Edge<V, E, W>(source, data.first, data.second, targets[j]));
}
return results;
}

template <typename V, typename E, typename W>
vector<Edge<V, E, W>>
AdjacencyListGraph<V, E, W>::getIncomingEdges(V destination) {
vector<Edge<V, E, W>> results;
vector<V> sources = edges.getKeys();
for (unsigned int i = 0; i < sources.size(); i++) {
if (edges.get(sources[i])->contains(destination)) {
pair<E, W> data = edges.get(sources[i])->get(destination);
results.push_back(
Edge<V, E, W>(sources[i], data.first, data.second, destination));
}
}
return results;
}

template <typename V, typename E, typename W>
vector<V> AdjacencyListGraph<V, E, W>::getNeighbors(V source) {
if (edges.contains(source)) {
return edges.get(source)->getKeys();
} else {
return vector<V>();
}
}
