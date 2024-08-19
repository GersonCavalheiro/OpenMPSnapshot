#pragma once

#include "data_point.h"
#include "knode.h"
#include "tree_printer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>


#define EMPTY_PLACEHOLDER std::numeric_limits<int>::min()


int powersum_of_two(array_size n, bool greater);


data_type *unpack_array(std::vector<DataPoint>::iterator first_point,
std::vector<DataPoint>::iterator last_point,
int n_components);


void unpack_array(data_type *dest, std::vector<DataPoint>::iterator first_point,
std::vector<DataPoint>::iterator last_point,
int n_components);


data_type *unpack_optional_array(std::optional<DataPoint> *array,
array_size n_datapoints, int n_components,
data_type fallback_value);


void merge_kd_trees(data_type *dest, data_type *branch1, data_type *branch2,
array_size branches_size, int n_components);

#ifdef ALTERNATIVE_SERIAL_WRITE
void rearrange_kd_tree(data_type *dest, data_type *src, array_size subtree_size,
int n_components);
#endif


KNode<data_type> *convert_to_knodes(data_type *tree, array_size n_datapoints,
int n_components,
array_size current_level_start,
array_size current_level_nodes,
array_size start_offset);


inline int select_splitting_dimension(int depth, int n_components) {
return depth % n_components;
}


array_size sort_and_split(DataPoint *array, array_size n_datapoints, int axis);


array_size sort_and_split(std::vector<DataPoint>::iterator first_data_point,
std::vector<DataPoint>::iterator end_data_point,
int axis);


inline std::vector<DataPoint>
as_data_points(data_type *data, array_size n_datapoints, int n_components) {
std::vector<DataPoint> data_points;
if (data == nullptr)
return data_points;

data_points.reserve(n_datapoints);
for (array_size i = 0; i < n_datapoints; i++) {
data_points.push_back(DataPoint(data + i * n_components));
}
return data_points;
}

#ifdef TEST

bool test_kd_tree(KNode<data_type> *root,
std::vector<std::optional<data_type>> *constraints,
int depth);
#endif
