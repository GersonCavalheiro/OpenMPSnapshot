#pragma once

#include "libfs.h"

#include "typedef_vcg.h"
#include "mesh_normals.h"
#include "mesh_coords.h"
#include "write_data.h"


#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cassert>
#include <limits>


#ifndef APPTAG
#define APPTAG "[cpp_geod] "
#endif



class Neighborhood {
public:
Neighborhood(size_t index, std::vector<std::vector<float>> coords, std::vector<float> distances, std::vector<std::vector<float>> normals) : index(index), coords(coords), distances(distances), normals(normals) {}
Neighborhood(size_t index, std::vector<std::vector<float>> coords, std::vector<float> distances) : index(index), coords(coords), distances(distances), normals(std::vector<std::vector<float>>(coords.size(), std::vector<float>(3, 0.0))) {}
Neighborhood(size_t index, std::vector<std::vector<float>> coords) : index(index), coords(coords), distances(std::vector<float>(coords.size(), 0.0)), normals(std::vector<std::vector<float>>(coords.size(), std::vector<float>(3, 0.0))) {}
Neighborhood() : index(0), coords(std::vector<std::vector<float>>(0)), distances(std::vector<float>(0)), normals(std::vector<std::vector<float>>(0)) {}

size_t index; 
std::vector<std::vector<float>> coords; 
std::vector<float> distances;  
std::vector<std::vector<float>> normals; 

size_t size() {
return this->distances.size();
}

std::vector<float> to_row(const size_t neigh_write_size, const float pvd=0.0, const bool use_pvd=false, const bool normals=true, const bool allow_nan=true) {
size_t row_length = 1 + ((3 + 1) * neigh_write_size); 
if(normals) {
row_length += 3 * neigh_write_size; 
}
if(use_pvd) {
row_length += 1; 
}
std::vector<float> row = std::vector<float>(row_length, 0.0);

size_t num_written = 0;
row[num_written] = (float)this->index; num_written++;


for(size_t j=0; j < neigh_write_size; j++) {  
if(j < this->size()) {
row[num_written] = this->coords[j][0]; num_written++;
row[num_written] = this->coords[j][1]; num_written++;
row[num_written] = this->coords[j][2]; num_written++;
} else {  
if(! allow_nan) {
return std::vector<float>();
}
row[num_written] = NAN; num_written++;
row[num_written] = NAN; num_written++;
row[num_written] = NAN; num_written++;
}
}

for(size_t j=0; j < neigh_write_size; j++) {  
if(j < this->size()) {
row[num_written] = this->distances[j]; num_written++;
} else {  
if(! allow_nan) {
return std::vector<float>();
}
row[num_written] = NAN; num_written++;
}
}

if(normals) {
for(size_t j=0; j < neigh_write_size; j++) {  
if(j < this->size()) {
row[num_written] = this->normals[j][0]; num_written++;
row[num_written] = this->normals[j][1]; num_written++;
row[num_written] = this->normals[j][2]; num_written++;
} else {  
if(! allow_nan) {
return std::vector<float>();
}
row[num_written] = NAN; num_written++;
row[num_written] = NAN; num_written++;
row[num_written] = NAN; num_written++;
}
}
}
if(use_pvd) {
row[num_written] = pvd; num_written++;
}
assert(num_written == row_length);
return(row);
}
};

std::vector<Neighborhood> neighborhoods_from_geod_neighbors(const std::vector<std::vector<GeodNeighbor> > geod_neighbors, MyMesh &mesh) {
size_t num_neighborhoods = geod_neighbors.size();
std::cout << std::string(APPTAG) << "Computing neighborhoods for " << num_neighborhoods << " vertices and their geodesic neighbors." << "\n";
std::vector<Neighborhood> neighborhoods;
size_t neigh_size;
std::vector<std::vector<float>> neigh_coords;
std::vector<std::vector<float>> neigh_normals;
std::vector<float> neigh_distances;
std::vector<float> source_vert_coords;
std::vector<int> neigh_indices;

std::vector<std::vector<float>> m_vnormals = mesh_vnormals(mesh);
std::vector<std::vector<float>> m_vcoords = mesh_vertex_coords(mesh);

size_t central_vert_mesh_idx;
size_t neigh_mesh_idx;
for(size_t i = 0; i < num_neighborhoods; i++) {
central_vert_mesh_idx = i;
neigh_size = geod_neighbors[i].size();
neigh_indices = std::vector<int>(neigh_size);
neigh_distances = std::vector<float>(neigh_size);
neigh_coords = std::vector<std::vector<float> >(neigh_size, std::vector<float> (3, 0.0));
neigh_normals = std::vector<std::vector<float> >(neigh_size, std::vector<float> (3, 0.0));
for(size_t j = 0; j < neigh_size; j++) {
neigh_mesh_idx = geod_neighbors[i][j].index; 
neigh_indices[j] = neigh_mesh_idx;
neigh_distances[j] = geod_neighbors[i][j].distance;  
neigh_coords[j] = std::vector<float> {m_vcoords[neigh_mesh_idx][0], m_vcoords[neigh_mesh_idx][1], m_vcoords[neigh_mesh_idx][2]};
source_vert_coords = std::vector<float> {m_vcoords[central_vert_mesh_idx][0], m_vcoords[central_vert_mesh_idx][1], m_vcoords[central_vert_mesh_idx][2]};
neigh_normals[j] = std::vector<float> {m_vnormals[neigh_mesh_idx][0], m_vnormals[neigh_mesh_idx][1], m_vnormals[neigh_mesh_idx][2]};
for(size_t k = 0; k < 3; k++) {
neigh_coords[j][k] -= source_vert_coords[k];
}
}
neighborhoods.push_back(Neighborhood(i, neigh_coords, neigh_distances, neigh_normals));
}
return neighborhoods;
}

std::vector<Neighborhood> neighborhoods_from_edge_neighbors(const std::vector<std::vector<int> > edge_neighbors, MyMesh &mesh, std::vector<bool> keep_verts = std::vector<bool>()) {

size_t num_neighborhoods = edge_neighbors.size();

if(keep_verts.empty()) {
keep_verts = std::vector<bool>(num_neighborhoods, true);
}
assert(keep_verts.size() == num_neighborhoods); 

std::vector<Neighborhood> neighborhoods;
size_t neigh_size, neigh_mesh_idx;
std::vector<std::vector<float>> neigh_coords;
std::vector<std::vector<float>> neigh_normals;
std::vector<float> neigh_distances;
std::vector<float> source_vert_coords;
std::vector<int> neigh_indices;

std::vector<std::vector<float>> m_vnormals = mesh_vnormals(mesh);
std::vector<std::vector<float>> m_vcoords = mesh_vertex_coords(mesh);

size_t central_vert_mesh_idx;
for(size_t i = 0; i < num_neighborhoods; i++) {
if(! keep_verts[i]) {
continue;
}
central_vert_mesh_idx = i;
neigh_size = edge_neighbors[i].size();
neigh_indices = std::vector<int>(neigh_size);
neigh_distances = std::vector<float>(neigh_size);
neigh_coords = std::vector<std::vector<float> >(neigh_size, std::vector<float> (3, 0.0));
neigh_normals = std::vector<std::vector<float> >(neigh_size, std::vector<float> (3, 0.0));
for(size_t j = 0; j < neigh_size; j++) {
neigh_mesh_idx = edge_neighbors[i][j];
neigh_indices[j] = neigh_mesh_idx;
neigh_coords[j] = std::vector<float> {m_vcoords[neigh_mesh_idx][0], m_vcoords[neigh_mesh_idx][1], m_vcoords[neigh_mesh_idx][2]};
neigh_normals[j] = std::vector<float> {m_vnormals[neigh_mesh_idx][0], m_vnormals[neigh_mesh_idx][1], m_vnormals[neigh_mesh_idx][2]};
source_vert_coords = std::vector<float> {m_vcoords[central_vert_mesh_idx][0], m_vcoords[central_vert_mesh_idx][1], m_vcoords[central_vert_mesh_idx][2]};
neigh_distances[j] = dist_euclid(neigh_coords[j], source_vert_coords); 
for(size_t k = 0; k < 3; k++) {
neigh_coords[j][k] -= source_vert_coords[k];
}
}
neighborhoods.push_back(Neighborhood(i, neigh_coords, neigh_distances, neigh_normals));
}
return neighborhoods;
}

std::string neighborhoods_to_json(std::vector<Neighborhood> neigh) {
std::stringstream is;
is << "{\n";
is << "  \"neighborhoods\": {\n";
Neighborhood nh;
for(size_t i=0; i < neigh.size(); i++) {
nh = neigh[i];
is << "  \"" << nh.index << "\": {\n";  
is << "    \"coords\": [\n";
is << "      ],\n";
is << "    \"distances\": [\n";
is << "      ],\n";
is << "    \"normals\": [\n";
is << "      ],\n";
is << "    }\n";
}
is << "  }\n";
is << "}\n";
is << "not implemented yet\n";
std::cerr << "neighborhoods_to_json: not implemented yet.\n";
return is.str();
}


std::string vec_to_csv_row(std::vector<float> vec) {
std::stringstream is;
for(size_t idx = 0; idx < vec.size(); idx++) {
is << vec[idx];
if(idx < vec.size() - 1) {
is << " ";
}
}
is << "\n";
return is.str();
}



std::string neighborhoods_to_csv(std::vector<Neighborhood> neigh, size_t neigh_write_size = 0, const bool allow_nan = false, const bool header=true, const bool normals = true, const std::string& input_pvd_file = "") {

std::vector<float> pvd;
if(! input_pvd_file.empty()) {
pvd = fs::read_curv_data(input_pvd_file);
}

size_t min_neighbor_count = (size_t)-1;  
size_t max_neighbor_count = 0; 
for(size_t i=0; i < neigh.size(); i++) {
if(neigh[i].size() < min_neighbor_count) {
min_neighbor_count = neigh[i].size();
}
if(neigh[i].size() > max_neighbor_count) {
max_neighbor_count = neigh[i].size();
}
}

if(neigh_write_size == 0) {
neigh_write_size = min_neighbor_count;
debug_print(CPP_GEOD_DEBUG_LVL_IMPORTANT, "Using auto-determined neighborhood size " + std::to_string(neigh_write_size) + " during Neighborhood CSV export.");
}

debug_print(CPP_GEOD_DEBUG_LVL_INFO, "Exporting " + std::to_string(neigh.size()) + " neighborhoods, with " + std::to_string(neigh_write_size) + " entries per neighborhood. Min neighborhood size = " + std::to_string(min_neighbor_count) + ", max = " + std::to_string(max_neighbor_count) + ".");

std::vector<int> failed_neighborhoods; 
for(size_t i=0; i < neigh.size(); i++) {
if(neigh[i].size() < neigh_write_size) {
failed_neighborhoods.push_back(i);
}
}
if(! allow_nan) {
if(failed_neighborhoods.size() >= 1) {
throw std::runtime_error("Failed to generate mesh neighborhood CSV representation:'" + std::to_string(failed_neighborhoods.size()) + " neighborhoods are smaller than neigh_write_size "  + std::to_string(neigh_write_size) + ", and allow_nan is false.\n");
}
} else {
std::cout << std::string(APPTAG) << "There are " << failed_neighborhoods.size() << " neighborhoods smaller than neigh_write_size " << neigh_write_size << ", will pad with 'NA' values.\n";
}

std::stringstream is;
if(header) {  
is << "source ";
for(size_t i=0; i < neigh_write_size; i++) { 
is << "n" << i << "cx" << " " << "n" << i << "cy" << " " << "n" << i << "cz";
if(i < neigh_write_size - 1) {
is << " ";
}
}
is << " ";
for(size_t i=0; i < neigh_write_size; i++) { 
is << "n" << i << "dist";
if(i < neigh_write_size - 1) {
is << " ";
}
}
if(normals) {
is << " ";
for(size_t i=0; i < neigh_write_size; i++) { 
is << "n" << i << "nx" << " " << "n" << i << "ny" << " " << "n" << i << "nz";
if(i < neigh_write_size - 1) {
is << " ";
}
}
}
if(! input_pvd_file.empty()) {  
is << " label";
}
is << "\n"; 
}

bool do_report = true;
if(do_report) {
float min_neigh_dist = std::numeric_limits<float>::max();
float max_neigh_dist = 0.0;
float dist_sum = 0.0;
size_t num_dists_considered = 0;
for(size_t i=0; i < neigh.size(); i++) {
for(size_t j=0; j < neigh_write_size; j++) {
if(j < neigh[i].size()) {
num_dists_considered++;
dist_sum += neigh[i].distances[j];
if(neigh[i].distances[j] < min_neigh_dist) {
min_neigh_dist = neigh[i].distances[j];
}
if(neigh[i].distances[j] > max_neigh_dist) {
max_neigh_dist = neigh[i].distances[j];
}
}
}
}
float mean_neigh_dist = dist_sum / (float)num_dists_considered;
std::cout << std::string(APPTAG) << "For exported neighborhoods (" << neigh_write_size << " entries max), the minimal distance is " << min_neigh_dist << ", mean is " << mean_neigh_dist << ", and max is " << max_neigh_dist << ".\n";
}

bool use_to_row = true;
if(use_to_row) {
std::vector<float> row;
for(size_t i=0; i < neigh.size(); i++) {
row = neigh[i].to_row(neigh_write_size, pvd[neigh[i].index], (! input_pvd_file.empty()), normals, allow_nan);
if(! row.empty()) { 
is << vec_to_csv_row(row);
}
}

} else {

for(size_t i=0; i < neigh.size(); i++) {

is << neigh[i].index;  
for(size_t j=0; j < neigh_write_size; j++) {  
if(j < neigh[i].size()) {
is << " " << neigh[i].coords[j][0] << " " << neigh[i].coords[j][1] << " " << neigh[i].coords[j][2];
} else {  
is << " NA NA NA";
}
}
for(size_t j=0; j < neigh_write_size; j++) {  
if(j < neigh[i].size()) {
is << " " << neigh[i].distances[j];
} else {  
is << " NA";
}
}
if(normals) {
for(size_t j=0; j < neigh_write_size; j++) {  
if(j < neigh[i].size()) {
is << " " << neigh[i].normals[j][0] << " " << neigh[i].normals[j][1] << " " << neigh[i].normals[j][2];
} else {  
is << " NA NA NA";
}
}
}
if(! input_pvd_file.empty()) {
is << " " << pvd[neigh[i].index];  
}
is << "\n";  
}
}
return is.str();
}


std::vector<std::vector<float>> neighborhoods_to_vvbin_mat(std::vector<Neighborhood> neigh, size_t neigh_write_size = 0, const bool allow_nan = false, const bool normals = true, const std::string& input_pvd_file = "") {
std::vector<float> pvd;
if(! input_pvd_file.empty()) {
pvd = fs::read_curv_data(input_pvd_file);
} else {
pvd = std::vector<float>(neigh.size(), 0.0); 
}

size_t min_neighbor_count = (size_t)-1;  
size_t max_neighbor_count = 0; 
for(size_t i=0; i < neigh.size(); i++) {
if(neigh[i].size() < min_neighbor_count) {
min_neighbor_count = neigh[i].size();
}
if(neigh[i].size() > max_neighbor_count) {
max_neighbor_count = neigh[i].size();
}
}

if(neigh_write_size == 0) {
neigh_write_size = min_neighbor_count;
debug_print(CPP_GEOD_DEBUG_LVL_INFO, "Using auto-determined neighborhood size " + std::to_string(neigh_write_size) + " during Neighborhood vvbin export.\n");
}

debug_print(CPP_GEOD_DEBUG_LVL_INFO, "Exporting " + std::to_string(neigh.size()) + " neighborhoods, with " + std::to_string(neigh_write_size) + " entries per neighborhood. Min neighborhood size = " + std::to_string(min_neighbor_count) + ", max = " + std::to_string(max_neighbor_count) + ".");

std::vector<int> failed_neighborhoods; 
for(size_t i=0; i < neigh.size(); i++) {
if(neigh[i].size() < neigh_write_size) {
failed_neighborhoods.push_back(i);
}
}
if(! allow_nan) {
if(failed_neighborhoods.size() >= 1) {
std::cout << std::string(APPTAG) << "There are " << std::to_string(failed_neighborhoods.size()) << " neighborhoods smaller than neigh_write_size "  << std::to_string(neigh_write_size) << ", and allow_nan is false, they will be filtered out.\n";
}
} else {
std::cout << std::string(APPTAG) << "There are " << failed_neighborhoods.size() << " neighborhoods smaller than neigh_write_size " << neigh_write_size << ", will pad with 'NA' values.\n";
}


std::vector<std::vector<float>> neigh_mat = std::vector<std::vector<float> >(0, std::vector<float> ());
std::vector<float> row;
for(size_t i=0; i < neigh.size(); i++) {
row = neigh[i].to_row(neigh_write_size, pvd[neigh[i].index], (! input_pvd_file.empty()), normals, allow_nan);
if(! row.empty()) { 
neigh_mat.push_back(row);
}
}
return neigh_mat;
}