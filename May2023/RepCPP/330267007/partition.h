

#pragma once

#include <iostream>
#include <vector>

#include "adjacencyListGraph.h"
#include "ccs_matrix.h"
#include "edge.h"
#include "stlHashTable.h"

using namespace std;


class Partition {
public:
Partition(){};
~Partition();

template <typename T> void from_lower_triangular_matrix(CCSMatrix<T> *matrix);

void clear();

void print();


vector<vector<int>> partitioning_get() { return partitioning; };

private:
vector<vector<int>> partitioning; 
};

template <typename T>
void Partition::from_lower_triangular_matrix(CCSMatrix<T> *matrix) {
AdjacencyListGraph<int, int, int>
dependency_graph; 
STLHashTable<int, int> num_parents_dict;
STLHashTable<int, bool> orphans;
for (int j = 0; j < matrix->num_col_get(); j++) {
dependency_graph.insertVertex(j);
num_parents_dict.insert(j, 0);
orphans.insert(j, true);
}
for (int j = 0; j < matrix->num_col_get(); j++) {
for (int p = matrix->column_pointer_get()[j];
p < matrix->column_pointer_get()[j + 1]; p++) {
int row_idx = matrix->row_index_get()[p];
if (row_idx > j && matrix->values_get()[p] != 0) {
dependency_graph.insertEdge(j, row_idx, 0, 0);
num_parents_dict.update(row_idx, num_parents_dict.get(row_idx) + 1);
if (orphans.contains(row_idx)) {
orphans.remove(row_idx);
}
}
}
}
unsigned int num_partitioned =
0; 
while (orphans.getSize() > 0) {
vector<int> partition = orphans.getKeys();
for (unsigned int i = 0; i < partition.size(); i++) {
int v = partition[i];
num_partitioned++;
orphans.remove(v);
vector<Edge<int, int, int>> outgoing_edges =
dependency_graph.getOutgoingEdges(v);
for (unsigned int j = 0; j < outgoing_edges.size(); j++) {
int child = outgoing_edges[j].target;
int new_num_parent = num_parents_dict.get(child) - 1;
num_parents_dict.update(child, new_num_parent);
if (new_num_parent <= 0) {
orphans.insert(child, true);
}
}
}
partitioning.push_back(partition);
}
if (num_partitioned != dependency_graph.getVertices().size()) {
throw runtime_error("Circular dependency found during partitioning, is the "
"matrix really lower triangular?");
}
}

Partition::~Partition() { this->clear(); }

void Partition::clear() {}

void Partition::print() {
for (unsigned int i = 0; i < partitioning.size(); i++) {
cout << "partition " << i << ": ";
for (unsigned int j = 0; j < partitioning[i].size(); j++) {
cout << partitioning[i][j] << " ";
}
cout << endl;
}
}