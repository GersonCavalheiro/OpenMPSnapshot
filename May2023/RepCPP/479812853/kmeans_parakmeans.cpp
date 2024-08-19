#include <iostream>
#include <vector>
#include <unordered_set>
#include <omp.h>
#include "utils/parse_args.h"
#include "utils/parse_input.h"
#include "utils/random.h"
#include "utils/kmeans_helper.h"


static inline std::vector<std::vector<double>> pick_random_centers(std::vector<std::vector<double>> &data, int k) {
std::vector<std::vector<double>> centers;
std::unordered_set<int> picked_centers;
for (int i = 0; i < k; i++) {
while (true) {
int index = (int) (random_double(0, 1) * (double) data.size());
if (index == (int) data.size()) {
index--;
}
if (picked_centers.find(index) == picked_centers.end()) {
picked_centers.insert(index);
centers.push_back(data[index]);
break;
}
}
}
return centers;
}

static inline int get_closest_center(std::vector<std::vector<double>> &centers, std::vector<double> &vec) {
int closest_center = 0;
double min_distance = pairwise_distance(centers[0], vec);
for (int i = 1; i < (int) centers.size(); i++) {
double distance = pairwise_distance(centers[i], vec);
if (distance < min_distance) {
min_distance = distance;
closest_center = i;
}
}
return closest_center;
}

static inline int
assign_points_to_centers(std::vector<std::vector<double>> &centers, std::vector<std::vector<double>> &data,
std::vector<int> &assignments, int start, int end) {
int changed = 0;
for (int i = start; i < end; i++) {
int new_assignment = get_closest_center(centers, data[i]);
if (new_assignment != assignments[i]) {
changed++;
assignments[i] = new_assignment;
}
}
return changed;
}

static inline void
recompute_centers(std::vector<std::vector<double>> &centers, std::vector<std::vector<double>> &data,
std::vector<int> &assignments, int data_start, int data_end) {
std::vector<int> center_counts = std::vector<int>(centers.size(), 0);
for (auto &center: centers) {
for (double &i: center) {
i = 0;
}
}
for (int i = data_start; i < data_end; i++) {
center_counts[assignments[i]]++;
for (int j = 0; j < (int) centers[assignments[i]].size(); j++) {
centers[assignments[i]][j] += data[i][j];
}
}
for (int i = 0; i < (int) centers.size(); i++) {
for (int j = 0; j < (int) centers[i].size(); j++) {
centers[i][j] /= center_counts[i];
}
}
}

static inline void add_local_centers_to_global_ones(std::vector<std::vector<double>> &global_centers,
std::vector<std::vector<double>> &local_centers) {
for (int i = 0; i < (int) local_centers.size(); i++) {
for (int j = 0; j < (int) local_centers[i].size(); j++) {
#pragma omp critical
global_centers[i][j] += local_centers[i][j];
}
}
}


static inline int
compute(std::vector<std::vector<double>> &global_centers, std::vector<std::vector<double>> &new_global_centers,
std::vector<int> &assignments, std::vector<std::vector<double>> &data, int data_start, int data_end) {
int changed = assign_points_to_centers(global_centers, data, assignments, data_start, data_end);
auto local_centers = std::vector<std::vector<double>>(global_centers.size(),
std::vector<double>(global_centers[0].size(), 0));
std::copy(global_centers.begin(), global_centers.end(), local_centers.begin());
recompute_centers(local_centers, data, assignments, data_start, data_end);
add_local_centers_to_global_ones(new_global_centers, local_centers);
return changed;
}

static inline void mean_global_centers(std::vector<std::vector<double>> &new_global_centers, int thread_count) {
for (auto &new_global_center: new_global_centers) {
for (double &j: new_global_center) {
j /= (double) thread_count;
}
}
}

int main(int argc, char **argv) {
ParsedArgs args = parse_args(argc, argv, "A basic sequential version of K-Means clustering");
auto data = parse_input(args.input_filename);
auto reference_assignments = parse_labels_input(args.labels_filename);
std::vector<int> assignments(data->size(), -1);

int cycle_no = 0;

double start_time, finish_time;
start_time = omp_get_wtime(); 

std::vector<std::vector<double>> centers = pick_random_centers(*data, args.k);
auto new_global_centers = std::vector<std::vector<double>>(args.k, std::vector<double>(
data->at(0).size(), 0));
int dataset_size_per_thread = (int) data->size() / args.thread_count;

omp_set_num_threads(args.thread_count);

while (true) {
cycle_no++;
new_global_centers = std::vector<std::vector<double>>(args.k, std::vector<double>(
data->at(0).size(), 0));
int changed = 0;
#pragma omp parallel for
for (int t = 0; t < args.thread_count; t++) {
int data_start = t * dataset_size_per_thread;
int data_end = (t + 1) * dataset_size_per_thread;
if (t == args.thread_count - 1) {
data_end = (int) data->size();
} else if (data_end > data->size()) {
data_end = (int) data->size();
}

changed += compute(centers, new_global_centers, assignments, *data, data_start, data_end);
}

std::cout << "Cycle #" << cycle_no << ": changed = " << changed << std::endl;
if (changed == 0) {
break;
}
mean_global_centers(new_global_centers, args.thread_count);
centers = new_global_centers;
}

finish_time = omp_get_wtime(); 

double nmi = compute_assignments_nmi(assignments, *reference_assignments, args.k);
std::cout << "NMI: " << nmi << std::endl;

std::cout << "Algorithm finished in: " << finish_time - start_time << " sec" << std::endl;

delete data;
delete reference_assignments;
return 0;
}
