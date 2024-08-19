#pragma once

#include <vector>
#include <iostream>
#include "edge.h"

using std::vector;
using std::ostream;


template <typename V, typename E, typename W>
class Graph {
public:
virtual ~Graph() {}


virtual vector<V> getVertices() = 0;


virtual void insertVertex(V v) = 0;


virtual void removeVertex(V v) = 0;


virtual bool containsVertex(V v) = 0;


virtual void insertEdge(V src, V dest, E label, W weight) = 0;


virtual void removeEdge(V src, V dest) = 0;


virtual bool containsEdge(V source, V destination) = 0;


virtual Edge<V,E,W> getEdge(V source, V destination) = 0;


virtual vector<Edge<V,E,W> > getEdges() = 0;


virtual vector<Edge<V,E,W> > getOutgoingEdges(V source) = 0;


virtual vector<Edge<V,E,W> > getIncomingEdges(V destination) = 0;


virtual vector<V> getNeighbors(V source) = 0;

public:
Graph() { }
private:
Graph(const Graph& other) = delete;
Graph& operator=(const Graph& other) = delete;
};

