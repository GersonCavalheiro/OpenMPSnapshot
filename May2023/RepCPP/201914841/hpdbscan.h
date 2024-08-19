

#ifndef HPDBSCAN_H
#define HPDBSCAN_H

#include <cmath>
#include <cstdint>
#include <hdf5.h>
#include <iomanip>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_OUTPUT
#include <iostream>
#endif

#include "atomic.h"
#include "constants.h"
#include "dataset.h"
#include "hdf5_util.h"
#include "io.h"
#include "rules.h"
#include "spatial_index.h"

class HPDBSCAN {
float m_epsilon;
size_t m_min_points;

#ifdef WITH_MPI
int m_rank;
int m_size;
#endif

template <typename T>
Rules local_dbscan(Clusters& clusters, const SpatialIndex<T>& index) {
const float EPS2 = m_epsilon * m_epsilon;
const size_t lower = index.lower_halo_bound();
const size_t upper = index.upper_halo_bound();

Rules rules;
Cell previous_cell = NOT_VISITED;
std::vector<size_t> neighboring_points;

#pragma omp parallel for schedule(dynamic, 32) private(neighboring_points) firstprivate(previous_cell) reduction(merge: rules)
for (size_t point = lower; point < upper; ++point) {
Cell current_cell = index.cell_of(point);

if (current_cell != previous_cell) {
neighboring_points = index.get_neighbors(current_cell);
previous_cell = current_cell;
}

std::vector<size_t> min_points_area;
Cluster cluster_label = NOISE;
if (neighboring_points.size() >= m_min_points) {
cluster_label = index.region_query(point, neighboring_points, EPS2, clusters, min_points_area);
}

if (min_points_area.size() >= m_min_points) {
atomic_min(clusters.data() + point, -cluster_label);

for (size_t other : min_points_area) {
Cluster other_cluster_label = std::abs(clusters[other]);
if (clusters[other] < 0) {
const std::pair<Cluster, Cluster> minmax = std::minmax(cluster_label, other_cluster_label);
rules.update(minmax.second, minmax.first);
}
atomic_min(clusters.data() + other, cluster_label);
}
}
else if (clusters[point] == NOT_VISITED) {
atomic_min(clusters.data() + point, NOISE);
}
}

return rules;
}

#ifdef WITH_MPI
template <typename T>
void merge_halos(Clusters& clusters, Rules& rules, const SpatialIndex<T>& index) {
Cuts cuts = index.compute_cuts();

int send_counts[m_size];
int recv_counts[m_size];
for (size_t i = 0; i < cuts.size(); ++i) {
send_counts[i] = static_cast<int>(cuts[i].second - cuts[i].first);
}
MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

int send_displs[m_size];
int recv_displs[m_size];
size_t total_items_to_receive = 0;
for (int i = 0; i < m_size; ++i) {
send_displs[i] = cuts[i].first;
recv_displs[i] = total_items_to_receive;
total_items_to_receive += static_cast<size_t>(recv_counts[i]);
}

const size_t upper_halo_bound = index.upper_halo_bound();
const size_t lower_halo_bound = index.lower_halo_bound();
Cluster halo_labels[total_items_to_receive];

MPI_Alltoallv(
clusters.data(), send_counts, send_displs, MPI_LONG,
halo_labels, recv_counts, recv_displs, MPI_LONG, MPI_COMM_WORLD
);

for (int i = 0; i < m_size; ++i) {
size_t offset = (i < m_rank ? lower_halo_bound : upper_halo_bound - recv_counts[i]);

for (int j = 0; j < recv_counts[i]; ++j) {
const size_t index = j + offset;
const Cluster own_cluster = clusters[index];
const Cluster halo_cluster = halo_labels[j + recv_displs[i]];

if (own_cluster < 0) {
const std::pair<Cluster, Cluster> minmax = std::minmax(std::abs(own_cluster), halo_cluster);
rules.update(minmax.second, minmax.first);
} else {
atomic_min(&clusters[index], halo_cluster);
}
}
}
}

void distribute_rules(Rules& rules) {
const int number_of_rules = static_cast<int>(rules.size());

int send_counts[m_size];
int send_displs[m_size];
int recv_counts[m_size];
int recv_displs[m_size];

for (int i = 0; i < m_size; ++i) {
send_counts[i] = 2 * number_of_rules;
send_displs[i] = 0;
}
MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

size_t total = 0;
for (int i = 0; i < m_size; ++i) {
recv_displs[i] = total;
total += recv_counts[i];
}

Cluster serialized_rules[send_counts[m_rank]];
size_t index = 0;
for (const auto& rule : rules) {
serialized_rules[index++] = rule.first;
serialized_rules[index++] = rule.second;
}

Cluster incoming_rules[total];
MPI_Alltoallv(
serialized_rules, send_counts, send_displs, MPI_LONG,
incoming_rules, recv_counts, recv_displs, MPI_LONG, MPI_COMM_WORLD
);
for (size_t i = 0; i < total; i += 2) {
rules.update(incoming_rules[i], incoming_rules[i + 1]);
}
}
#endif

void apply_rules(Clusters& clusters, const Rules& rules) {
#pragma omp parallel for
for (size_t i = 0; i < clusters.size(); ++i) {
const bool is_core = clusters[i] < 0;
Cluster cluster = std::abs(clusters[i]);
Cluster matching_rule = rules.rule(cluster);

while (matching_rule < NOISE) {
cluster = matching_rule;
matching_rule = rules.rule(matching_rule);
}
clusters[i] = is_core ? -cluster : cluster;
}
}

#ifdef WITH_OUTPUT
void summarize(const Dataset& dataset, const Clusters& clusters) const {
std::unordered_set<Cluster> unique_clusters;
size_t cluster_points = 0;
size_t core_points = 0;
size_t noise_points = 0;

for (size_t i = 0; i < dataset.m_chunk[0]; ++i) {
const Cluster cluster = clusters[i];
unique_clusters.insert(std::abs(cluster));

if (cluster == 0) {
++noise_points;
} else {
++cluster_points;
}
if (cluster < 0) {
++core_points;
}
}
size_t metrics[] = {cluster_points, noise_points, core_points};

#ifdef WITH_MPI
int number_of_unique_clusters = static_cast<int>(unique_clusters.size());
int set_counts[m_size];
int set_displs[m_size];

if (m_rank == 0) {
#endif
std::cout << "Summary..." << std::endl;
#ifdef WITH_MPI
}
MPI_Gather(&number_of_unique_clusters, 1, MPI_INT, set_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

Clusters global_buffer;
Clusters local_buffer(number_of_unique_clusters);
std::copy(unique_clusters.begin(), unique_clusters.end(), local_buffer.begin());

size_t buffer_size = 0;
if (m_rank == 0) {
for (int i = 0; i < m_size; ++i) {
set_displs[i] = buffer_size;
buffer_size += set_counts[i];
}
global_buffer.resize(buffer_size);
}

MPI_Gatherv(
local_buffer.data(), number_of_unique_clusters, MPI_LONG,
global_buffer.data(), set_counts, set_displs, MPI_LONG, 0, MPI_COMM_WORLD
);
MPI_Reduce(
m_rank == 0 ? MPI_IN_PLACE : metrics, metrics, sizeof(metrics) / sizeof(size_t),
MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD
);

if (m_rank == 0) {
unique_clusters.insert(global_buffer.begin(), global_buffer.end());
#endif
std::cout << "\tClusters:       " << (metrics[1] ? unique_clusters.size() - 1 : unique_clusters.size()) << std::endl
<< "\tCluster points: " << metrics[0] << std::endl
<< "\tNoise points:   " << metrics[1] << std::endl
<< "\tCore points:    " << metrics[2] << std::endl;
#ifdef WITH_MPI
}
#endif
}
#endif

public:
HPDBSCAN(float epsilon, size_t min_points) : m_epsilon(epsilon), m_min_points(min_points) {
#ifdef WITH_MPI
MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
MPI_Comm_size(MPI_COMM_WORLD, &m_size);
#endif

if (epsilon <= 0.0) {
throw std::invalid_argument("epsilon needs to be positive");
}
}

Clusters cluster(const std::string& path, const std::string& dataset) {
return cluster(path, dataset, omp_get_max_threads());
}

Clusters cluster(const std::string& path, const std::string& dataset, int threads) {
Dataset data = IO::read_hdf5(path, dataset);

H5T_class_t type_class = H5Tget_class(data.m_type);
size_t precision = H5Tget_precision(data.m_type);

if (type_class == H5T_INTEGER) {
H5T_sign_t sign = H5Tget_sign(data.m_type);

if (sign == H5T_SGN_2) {
if (precision == 8) {
return cluster<int8_t>(data, threads);
} else if (precision == 16) {
return cluster<int16_t>(data, threads);
} else if (precision == 32) {
return cluster<int32_t>(data, threads);
} else if (precision == 64) {
return cluster<int64_t>(data, threads);
} else {
throw std::invalid_argument("Unsupported signed integer precision");
}
} else {
if (precision == 8) {
return cluster<uint8_t>(data, threads);
} else if (precision == 16) {
return cluster<uint16_t>(data, threads);
} else if (precision == 32) {
return cluster<uint32_t>(data, threads);
} else if (precision == 64) {
return cluster<uint64_t>(data, threads);
} else {
throw std::invalid_argument("Unsupported unsigned integer precision");
}
}
} else if (type_class == H5T_FLOAT) {
if (precision == 32) {
return cluster<float>(data, threads);
} else if (precision == 64) {
return cluster<double>(data, threads);
} else {
throw std::invalid_argument("Unsupported floating point precision");
}
} else {
throw std::invalid_argument("Unsupported data set type");
}
}

template <typename T>
Clusters cluster(Dataset& dataset, int threads=omp_get_max_threads()) {
#ifdef WITH_OUTPUT
double execution_start = omp_get_wtime();
#endif
omp_set_num_threads(threads);

#ifdef WITH_OUTPUT
std::cout << std::fixed << std::setw(11) << std::setprecision(6) << std::setfill(' ');
#endif

SpatialIndex<T> index(dataset, m_epsilon);
Clusters clusters(dataset.m_chunk[0], NOT_VISITED);

#ifdef WITH_OUTPUT
double start = omp_get_wtime();

#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "Clustering..." << std::endl;
std::cout << "\tDBSCAN...              " << std::flush;
#ifdef WITH_MPI
}
#endif
start = omp_get_wtime();
#endif
Rules rules = local_dbscan(clusters, index);
#ifdef WITH_OUTPUT
#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
#ifdef WITH_MPI
}
#endif
start = omp_get_wtime();
#endif

#ifdef WITH_MPI
#ifdef WITH_OUTPUT
if (m_rank == 0) {
std::cout << "\tMerging halos...       " << std::flush;
}
#endif
merge_halos(clusters, rules, index);
distribute_rules(rules);
#ifdef WITH_OUTPUT
if (m_rank == 0) {
std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
start = omp_get_wtime();
}
#endif
#endif

#ifdef WITH_OUTPUT
#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "\tAppyling rules...      " << std::flush;
#ifdef WITH_MPI
}
#endif
#endif
apply_rules(clusters, rules);
#ifdef WITH_OUTPUT
#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
#ifdef WITH_MPI
}
#endif
#endif

#ifdef WITH_OUTPUT
#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "\tRecovering order...    " << std::flush;
start = omp_get_wtime();
#ifdef WITH_MPI
}
#endif
#endif
index.recover_initial_order(clusters);
#ifdef WITH_OUTPUT
#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "[OK] in " << omp_get_wtime() - start << std::endl;
#ifdef WITH_MPI
}
#endif
start = omp_get_wtime();
#endif

#ifdef WITH_OUTPUT
summarize(dataset, clusters);
#ifdef WITH_MPI
if (m_rank == 0) {
#endif
std::cout << "Total time: " << omp_get_wtime() - execution_start << std::endl;
#ifdef WITH_MPI
}
#endif
#endif

return clusters;
}

template <typename T>
Clusters cluster(T* data, int dim0, int dim1, int threads) {
hsize_t chunk[2] = {static_cast<hsize_t>(dim0), static_cast<hsize_t>(dim1)};
Dataset dataset(data, chunk, HDF5_Types<T>::map());

return cluster<T>(dataset, threads);
}

template <typename T>
Clusters cluster(T* data, int dim0, int dim1) {
return cluster(data, dim0, dim1, omp_get_max_threads());
}
};

#endif 
