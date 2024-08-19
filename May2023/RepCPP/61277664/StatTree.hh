#pragma once

#include <vector>
#include "VertexSet.h"


class StatTree {
private:
std::vector<vidType> vertices;
std::vector<StatTree> subtrees;
vidType full_size;
public:
StatTree() {}
StatTree(vidType full_size): full_size(full_size) {}

vidType get_full_size() const { return full_size; }
vidType size() const { return vertices.size(); }
bool is_leaf() const { return subtrees.empty(); }

std::vector<size_t> full_size_trace(size_t base_size) {
size_t current_size = base_size + full_size;
std::vector<size_t> trace {current_size};

for (auto &t : subtrees) {
if (!t.is_leaf()) {
std::vector<size_t> sub_trace = t.full_size_trace(current_size);
trace.insert(trace.end(), sub_trace.begin(), sub_trace.end());
}
}

return trace;
}

std::vector<size_t> size_trace(size_t base_size) {
size_t current_size = base_size + vertices.size();
std::vector<size_t> trace {current_size};

for (auto &t : subtrees) {
if (!t.is_leaf()) {
std::vector<size_t> sub_trace = t.size_trace(current_size);
trace.insert(trace.end(), sub_trace.begin(), sub_trace.end());
}
}

return trace;
}

void set_full_size(vidType full_size) {
this->full_size = full_size;
}

void add_vertex(vidType vertex) {
vertices.push_back(vertex);
}

StatTree& add_subtree(StatTree&& subtree) {
subtrees.push_back(subtree);
return subtrees.back();
}

void clear() {
full_size = 0;
vertices.clear();
for (auto &t : subtrees) {
t.clear();
}
subtrees.clear();
}

};

