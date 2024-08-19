#pragma once

#include "data_point.h"
#include "knode.h"
#include "process_utils.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <math.h>
#include <optional>
#include <unistd.h>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#else
#include <omp.h>
#endif


#define TAG_RIGHT_PROCESS_PROCESSING_OVER 10

#define TAG_RIGHT_PROCESS_N_ITEMS 11

#define TAG_RIGHT_PROCESS_START_INFO 12

#define TAG_RIGHT_PROCESS_START_DATA 13

class KDTreeGreenhouse {
private:
array_size n_datapoints;
int n_components;

int starting_depth = 0;

#ifdef USE_MPI
int rank = -1;

MPI_Comm no_main_communicator;
#endif

int n_parallel_workers = -1;

int max_parallel_depth = 0;

int surplus_workers = 0;

array_size tree_size = 0;

std::optional<DataPoint> *growing_tree = nullptr;

#ifdef USE_MPI
int parent = -1;

std::vector<DataPoint> parallel_splits;

std::vector<int> children;

data_type *right_branch_memory_pool = nullptr;
MPI_Request right_branch_send_data_request = MPI_REQUEST_NULL;
#endif

array_size grown_kdtree_size = 0;
KNode<data_type> *grown_kd_tree = nullptr;

data_type *grow_kd_tree(std::vector<DataPoint> &data_points);

#ifdef USE_MPI
data_type *retrieve_dataset_info();
void build_tree_parallel(std::vector<DataPoint>::iterator first_data_point,
std::vector<DataPoint>::iterator end_data_point,
int depth);
#endif

void build_tree_single_core(std::vector<DataPoint>::iterator first_data_point,
std::vector<DataPoint>::iterator end_data_point,
int depth, array_size region_width,
array_size region_start_index,
array_size branch_starting_index);
data_type *finalize();

public:
KDTreeGreenhouse(data_type *data, array_size n_datapoints, int n_components);
~KDTreeGreenhouse() { delete grown_kd_tree; }

KNode<data_type> &&extract_grown_kdtree() {
return std::move(*grown_kd_tree);
}
array_size get_grown_kdtree_size() { return grown_kdtree_size; }
};
