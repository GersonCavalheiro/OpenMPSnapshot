#pragma once

#include "knode.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>


data_type *read_file(const std::string &filename, std::size_t *size, int *dims);


template <typename T>
void write_file(const std::string &filename, KNode<T> *root, int dims) {
std::ofstream outdata;
outdata.open(filename, std::fstream::out);
if (!outdata) {
throw std::invalid_argument("File not found.");
}

std::queue<KNode<T> *> to_visit;
to_visit.push(root);

while (to_visit.size() > 0) {
KNode<T> *node = to_visit.front();
to_visit.pop();

for (int i = 0; i < dims; ++i) {
outdata << node->get_data(i);
if (i < dims - 1)
outdata << ",";
}

if (node->get_left() != nullptr)
to_visit.push(node->get_left());
if (node->get_right() != nullptr)
to_visit.push(node->get_right());

outdata << std::endl;
}
}
