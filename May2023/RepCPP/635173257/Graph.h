#pragma once

#include <fstream>
#include <iostream>

#include "Edge.cpp"

using namespace std;

struct Graph {
int vertexesCount;
int edgesSize;
int edgesAllocatedSize;
Edge *edges;

Graph();

Graph(const Graph &source);

~Graph();

Graph &operator=(const Graph &source);

bool loadFromFile(const string &path);

void addEdge(Edge edge);

friend ostream &operator<<(ostream &os, const Graph &graph);
};