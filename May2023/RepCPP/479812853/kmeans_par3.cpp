#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_set>
#include <omp.h>
#include <algorithm>
#include <random>
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
std::vector<int> &assignments, ParsedArgs args) {
int changed = 0;
#pragma omp parallel for num_threads(args.thread_count)
for (int i = 0; i < (int) data.size(); i++) {
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
std::vector<int> &assignments) {
std::vector<int> center_counts = std::vector<int>(centers.size(), 0);
for (auto &center: centers) {
for (double &i: center) {
i = 0;
}
}
for (int i = 0; i < (int) data.size(); i++) {
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

class SetOfMeans
{
public:
int k;
double intracluster_distance = 0.0;  
double intercluster_distance = 0.0;  
double rank;
ParsedArgs args;
std::vector<std::vector<double>> data;
std::vector<int> assignments;
std::vector<std::vector<double>> centers;
std::vector<std::vector<double>> org_centers;



SetOfMeans(std::vector<std::vector<double>> inputdata, int k, ParsedArgs argsinput){
data = inputdata;
assignments.resize(data.size(), -1);
centers = pick_random_centers(data, k);
org_centers = centers;
args = argsinput;
}

void process_iterations(){
for(int i=0; i < 5; i++){
assign_points_to_centers(centers, data, assignments, args);
recompute_centers(centers, data, assignments);
}
calculate_distance();
}

void calculate_distance(){
for(int i=0; i < assignments.size(); i++){
intracluster_distance += pairwise_distance(data[i], centers[assignments[i]]);
}

for(int i=0; i < centers.size(); i++){
for(int j = i; j < centers.size(); j++){
intercluster_distance += pairwise_distance(centers[i], centers[j]);
}
}

if(std::isnan(intercluster_distance)){
intercluster_distance = 0;
}
}

};

bool compByIntracluster(SetOfMeans* a, SetOfMeans* b)
{
return a->intracluster_distance < b->intracluster_distance;
}

bool compByIntercluster(SetOfMeans* a, SetOfMeans* b)
{
return a->intercluster_distance > b->intercluster_distance;
}

bool compByRank(SetOfMeans* a, SetOfMeans* b)
{
return a->rank < b->rank;
}

int main(int argc, char **argv) {
ParsedArgs args = parse_args(argc, argv, "A basic sequential version of K-Means clustering");
auto data = parse_input(args.input_filename);
auto reference_assignments = parse_labels_input(args.labels_filename);
std::vector<int> assignments(data->size(), -1);

auto shuffled_data = *data;
auto rng = std::default_random_engine {};
std::shuffle(std::begin(shuffled_data), std::end(shuffled_data), rng);
std::vector<std::vector<double>> datasubset;
for(int i = 0; i < sqrt(data->size()); i++){
datasubset.push_back(data->at(i));
}

double start_time, finish_time;
start_time = omp_get_wtime(); 
std::vector<SetOfMeans*> setofmeans_vec;
for(int i=0; i < 500; i++){
SetOfMeans* current = new SetOfMeans(datasubset, args.k, args);
setofmeans_vec.push_back(current);
}


#pragma omp parallel for num_threads(args.thread_count)
for(int i=0; i < setofmeans_vec.size(); i++){
setofmeans_vec[i]->process_iterations();
}

std::sort(setofmeans_vec.begin(), setofmeans_vec.end(), compByIntracluster);
for(int i = 0; i < setofmeans_vec.size(); i++){
setofmeans_vec[i]->rank = i;
}

std::sort(setofmeans_vec.begin(), setofmeans_vec.end(), compByIntercluster);
for(int i = 0; i < setofmeans_vec.size(); i++){
setofmeans_vec[i]->rank += 0.5 *i;
}

int cycle_no = 0;

std::vector<std::vector<double>> centers = (*std::min_element(setofmeans_vec.begin(), setofmeans_vec.end(), compByRank))->org_centers;

for(int i = 0; i < centers.size(); i++){
std::cout<< centers[i][0]<<" "<< centers[i][1] << " "<< centers[i][2]<< std::endl;
}
while (true) {
cycle_no++;
int changed = assign_points_to_centers(centers, *data, assignments, args);
std::cout << "Cycle #" << cycle_no << ": changed = " << changed << std::endl;
if (changed == 0) {
break;
}
recompute_centers(centers, *data, assignments);
}

finish_time = omp_get_wtime(); 

double nmi = compute_assignments_nmi(assignments, *reference_assignments, args.k);
std::cout << "NMI: " << nmi << std::endl;

std::cout << "Algorithm finished in: " << finish_time - start_time << " sec" << std::endl;

delete data;
delete reference_assignments;
return 0;
}
