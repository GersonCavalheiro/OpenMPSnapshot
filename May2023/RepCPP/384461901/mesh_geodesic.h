#pragma once

#include "libfs.h"
#include "spline.h"

#include "typedef_vcg.h"
#include "mesh_area.h"
#include "mesh_edges.h"
#include "vec_math.h"

#include <vcg/complex/complex.h>
#include <vcg/complex/append.h>
#include <vcg/complex/algorithms/geodesic.h>
#include <vcg/container/simple_temporary_data.h>

#define _USE_MATH_DEFINES
#include <cmath>


#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cassert>


std::vector<float> geodist(MyMesh& m, std::vector<int> verts, float maxdist) {

int n = verts.size();
VertexIterator vi;
FaceIterator fi;

m.vert.EnableVFAdjacency();
m.vert.EnableQuality();
m.face.EnableFFAdjacency();
m.face.EnableVFAdjacency();
tri::UpdateTopology<MyMesh>::VertexFace(m);

std::vector<MyVertex*> seedVec(verts.size());
for (int i=0; i < n; i++) {
vi = m.vert.begin()+verts[i];
seedVec[i] = &*vi;
}

tri::EuclideanDistance<MyMesh> ed;
if(maxdist > 0.0) {
tri::Geodesic<MyMesh>::PerVertexDijkstraCompute(m,seedVec,ed,maxdist);
} else {
tri::Geodesic<MyMesh>::PerVertexDijkstraCompute(m,seedVec,ed);
}

std::vector<float> geodists(m.vn);
vi=m.vert.begin();
for (int i=0; i < m.vn; i++) {
geodists[i] = vi->Q();
++vi;
}
return geodists;
}


std::vector<float> mean_geodist_p(MyMesh &m) {

fs::Mesh surf;
fs_surface_from_vcgmesh(&surf, m);

std::vector<float> meandists;
size_t nv = surf.num_vertices();
float max_dist = -1.0;
meandists.resize(nv);

# pragma omp parallel for firstprivate(max_dist) shared(surf, meandists)
for(size_t i=0; i<nv; i++) {
MyMesh m;
vcgmesh_from_fs_surface(&m, surf);
std::vector<int> query_vert;
query_vert.resize(1);
query_vert[0] = i;
std::vector<float> gdists = geodist(m, query_vert, max_dist);
double dist_sum = 0.0;
for(size_t j=0; j<nv; j++) {
dist_sum += gdists[j];
}
meandists[i] = (float)(dist_sum / nv);
}
return meandists;
}

struct GeodNeighbor {
GeodNeighbor() : index(0), distance(0.0) {}
GeodNeighbor(size_t index, float distance) : index(index), distance(distance) {}
size_t index; 
float distance; 
std::vector<float> normals; 
};



std::vector<std::vector<GeodNeighbor>> geod_neighborhood(MyMesh &m, const float max_dist = 5.0, const bool include_self = true) {

fs::Mesh surf;
fs_surface_from_vcgmesh(&surf, m);

size_t nv = surf.num_vertices();
std::vector<std::vector<GeodNeighbor>> neighborhoods(nv, std::vector<GeodNeighbor>());

# pragma omp parallel for firstprivate(max_dist) shared(surf, neighborhoods)
for(size_t i=0; i<nv; i++) {
MyMesh m;
vcgmesh_from_fs_surface(&m, surf);
std::vector<int> query_vert= {(int)i};
std::vector<float> gdists = geodist(m, query_vert, max_dist);

for(size_t j=0; j<gdists.size(); j++) {
if(i == j) {
if(include_self) {
neighborhoods[i].push_back(GeodNeighbor(j, 0.0)); 
}
} else {
if(gdists[j] > 0.0 && gdists[j] <= max_dist) {
neighborhoods[i].push_back(GeodNeighbor(j, gdists[j]));
}
}
}

}
return neighborhoods;
}


std::string geod_neigh_to_json(std::vector<std::vector<GeodNeighbor>> neigh) {
std::stringstream is;
is << "{\n";
is << "  \"neighbors\": {\n";
for(size_t i=0; i < neigh.size(); i++) {
is << "  \"" << i << "\": [";
for(size_t j=0; j < neigh[i].size(); j++) {
is << " " << neigh[i][j].index;
if(j < neigh[i].size()-1) {
is << ",";
}
}
is << " ]";
if(i < neigh.size()-1) {
is <<",";
}
is <<"\n";
}
is << "  },\n";
is << "  \"distances\": {\n";
for(size_t i=0; i < neigh.size(); i++) {
is << "  \"" << i << "\": [";
for(size_t j=0; j < neigh[i].size(); j++) {
is << " " << neigh[i][j].distance;
if(j < neigh[i].size()-1) {
is << ",";
}
}
is << " ]";
if(i < neigh.size()-1) {
is <<",";
}
is <<"\n";
}
is << "  }\n";
is << "}\n";
return is.str();
}


std::string geod_neigh_to_csv(const std::vector<std::vector<GeodNeighbor>> neigh, const std::string sep=",") {
std::stringstream is;
is << "source" << sep << "target" << sep << "distance" << "\n";

for(size_t i=0; i < neigh.size(); i++) {
for(size_t j=0; j < neigh[i].size(); j++) {
is << i << sep << neigh[i][j].index << sep << neigh[i][j].distance << "\n";
}
}
return is.str();
}


std::vector<float> mean_geodist(MyMesh &m) {
std::vector<float> meandists;
size_t nv = m.VN();
float max_dist = -1.0;
meandists.resize(nv);

for(size_t i=0; i<nv; i++) {
std::vector<int> query_vert;
query_vert.resize(1);
query_vert[0] = i;
std::vector<float> gdists = geodist(m, query_vert, max_dist);
double dist_sum = 0.0;
for(size_t j=0; j<nv; j++) {
dist_sum += gdists[j];
}
meandists[i] = (float)(dist_sum / nv);
}
return meandists;
}


template<typename T>
int numsteps_for_stepsize(T start_in, T end_in, double stepsize) {
double start = static_cast<double>(start_in);
double end = static_cast<double>(end_in);
double delta = end - start;
int numsteps = (int)ceil(delta / stepsize);
return numsteps + 1;
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in) {

std::vector<double> linspaced;

double start = static_cast<double>(start_in);
double end = static_cast<double>(end_in);
double num = static_cast<double>(num_in);

if (num == 0) { return linspaced; }
if (num == 1) {
linspaced.push_back(start);
return linspaced;
}

double delta = (end - start) / (num - 1);

for(int i=0; i < num-1; ++i) {
linspaced.push_back(start + delta * i);
}
linspaced.push_back(end); 
return linspaced;
}


std::vector<std::vector<double>> _compute_geodesic_circle_stats(MyMesh& m, std::vector<float> geodist, std::vector<double> sample_at_radii) {

fs::Mesh surf;
fs_surface_from_vcgmesh(&surf, m);
float max_possible_float = std::numeric_limits<float>::max();

int nr = sample_at_radii.size();
std::vector<double> per_face_area = mesh_area_per_face(m);

std::vector<double> areas_by_radius(nr);
std::vector<double> perimeters_by_radius(nr);

for(int radius_idx=0; radius_idx<nr; radius_idx++) {
double radius = sample_at_radii[radius_idx];

std::vector<bool> vert_in_radius(m.vn); 
for(int i=0; i<m.vn; i++) {
vert_in_radius[i] = geodist[i] < radius;
}

std::vector<int> faces_num_verts_in_radius(m.fn);
for(int i=0; i<m.fn; i++) {
faces_num_verts_in_radius[i] = 0;
for(int j=0; j<3; j++) {
if(geodist[surf.fm_at(i, j)] < radius) {
faces_num_verts_in_radius[i]++;
}
}
}

double total_area_in_radius = 0.0; 
double total_perimeter = 0.0;

for(int i=0; i<m.fn; i++) {
if(faces_num_verts_in_radius[i] == 3) {
total_area_in_radius += per_face_area[i];
}
}

for(int i=0; i<m.fn; i++) {
if(faces_num_verts_in_radius[i] != 3 && faces_num_verts_in_radius[i] != 0) {
int num_verts_in_radius = faces_num_verts_in_radius[i];
int k = -1;
std::vector<int> face_verts = surf.face_vertices(i);
if(num_verts_in_radius == 2) { 
for(int j=0; j<3; j++) {
if(vert_in_radius[face_verts[j]] == false) {
k=j;
}
}
} else { 
for(int j=0; j<3; j++) {
if(vert_in_radius[face_verts[j]] == true) {
k=j;
}
}
}
assert(k>=0);
std::vector<int> face_verts_copy = surf.face_vertices(i); 
if(k == 1) {
face_verts[0] = face_verts_copy[1];
face_verts[1] = face_verts_copy[2];
face_verts[2] = face_verts_copy[0];
} else if(k==2) {
face_verts[0] = face_verts_copy[2];
face_verts[1] = face_verts_copy[0];
face_verts[2] = face_verts_copy[1];
} 

std::vector<float> face_vertex_dists(3);  
face_vertex_dists[0] = geodist[face_verts[0]] - radius;
face_vertex_dists[1] = geodist[face_verts[1]] - radius;
face_vertex_dists[2] = geodist[face_verts[2]] - radius;

assert(geodist[face_verts[0]] < (max_possible_float - 0.01));
assert(geodist[face_verts[1]] < (max_possible_float - 0.01));
assert(geodist[face_verts[2]] < (max_possible_float - 0.01));

std::vector<float> coords_v0 = surf.vertex_coords(face_verts[0]);
std::vector<float> coords_v1 = surf.vertex_coords(face_verts[1]);
std::vector<float> coords_v2 = surf.vertex_coords(face_verts[2]);

float alpha1 = face_vertex_dists[1]/(face_vertex_dists[1]-face_vertex_dists[0]);
std::vector<float> v1 = alpha1 * coords_v0 + (1.0f-alpha1) * coords_v1;
float alpha2 = face_vertex_dists[2]/(face_vertex_dists[2]-face_vertex_dists[0]);
std::vector<float> v2 = alpha2 * coords_v0 + (1.0f-alpha2) * coords_v2;

float b = vnorm(cross(coords_v0 - v1, coords_v0 - v2)) / 2.0;
if(num_verts_in_radius == 2) { 
total_area_in_radius += per_face_area[i] - b;
} else { 
total_area_in_radius += b;
}

total_perimeter += vnorm(v1 - v2);
}

}
areas_by_radius[radius_idx] = total_area_in_radius;
perimeters_by_radius[radius_idx] = total_perimeter;
}

std::vector<std::vector<double>> res;
res.push_back(areas_by_radius);
res.push_back(perimeters_by_radius);
return res;
}


std::vector<std::vector<float>> geodesic_circles(MyMesh& m, std::vector<int> query_vertices, float scale=5.0, bool do_meandist=false) {

double sampling = 10.0;
double mesh_area = mesh_area_total(m);
double area_scale = (scale * mesh_area) / 100.0;
double r_cycle = sqrt(area_scale / M_PI);
float max_possible_float = std::numeric_limits<float>::max();

std::vector<double> edge_lengths = mesh_edge_lengths(m);
double mean_len = std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0) / (double)edge_lengths.size();
double max_edge_len = *std::max_element(edge_lengths.begin(), edge_lengths.end());
std::cout  << "     o Mesh has " << edge_lengths.size() << " edges with average length " << mean_len << " and maximal length " << max_edge_len << ".\n";

double extra_dist = max_edge_len * 8.0;
double max_dist = r_cycle + extra_dist; 
if(do_meandist) {
max_dist = -1.0; 
} else {
std::cout  << "     o Using extra_dist=" << extra_dist << ", resulting in max_dist=" << max_dist << ".\n";
}

if(query_vertices.empty()) {
query_vertices.resize(m.vn);
for(int i=0; i<m.vn; i++) {
query_vertices[i] = i;
}
}

std::vector<float> radius, perimeter, meandist;
size_t nqv = query_vertices.size();
radius.resize(nqv);
perimeter.resize(nqv);
meandist.resize(nqv);

fs::Mesh surf;  
fs_surface_from_vcgmesh(&surf, m);

# pragma omp parallel for shared(surf, radius, perimeter, meandist)
for(size_t i=0; i<nqv; i++) {
MyMesh mt; 
vcgmesh_from_fs_surface(&mt, surf);
int qv = query_vertices[i];
std::vector<int> query_vertex = { qv };
std::vector<float> v_geodist = geodist(mt, query_vertex, max_dist);

if(do_meandist) {
meandist[i] = std::accumulate(v_geodist.begin(), v_geodist.end(), 0.0) / (float)v_geodist.size();
} else {
for(size_t j=0; j<v_geodist.size(); j++) {
if(j != (size_t)qv) { 
if(v_geodist[j] <= 0.000000001) {
v_geodist[j] = max_possible_float;
}
}
}
}

std::vector<double> sample_at_radii = linspace<double>(r_cycle-10.0, r_cycle+10.0, sampling);
std::vector<std::vector<double>> circle_stats = _compute_geodesic_circle_stats(m, v_geodist, sample_at_radii);
std::vector<double> circle_areas = circle_stats[0];
std::vector<double> circle_perimeters = circle_stats[1];

assert(sample_at_radii.size() == circle_areas.size());
assert(sample_at_radii.size() == circle_perimeters.size());

std::vector<double> x = linspace<double>(1.0, sampling, numsteps_for_stepsize(1.0, sampling, 1.0)); 
std::vector<double> xx = linspace<double>(1.0, sampling, numsteps_for_stepsize(1.0, sampling, 0.1));  
int num_samples = xx.size();

assert(x.size() == circle_areas.size()); 

tk::spline spl_areas(x, circle_areas);
tk::spline spl_radius(x, sample_at_radii);
tk::spline spl_perimeters(x, circle_perimeters);
std::vector<double> sampled_areas(num_samples);
for(int i=0; i<num_samples; i++) { sampled_areas[i] = spl_areas(xx[i]); }
std::vector<double> sampled_radii(num_samples);
for(int i=0; i<num_samples; i++) { sampled_radii[i] = spl_radius(xx[i]); }
std::vector<double> sampled_perimeters(num_samples);
for(int i=0; i<num_samples; i++) { sampled_perimeters[i] = spl_perimeters(xx[i]); }

for(int i=0; i<num_samples; i++) {
sampled_areas[i] = fabs(area_scale - sampled_areas[i]);
}
int min_index = std::distance(sampled_areas.begin(),std::min_element(sampled_areas.begin(),sampled_areas.end()));
radius[i] = sampled_radii[min_index];
perimeter[i] = sampled_perimeters[min_index];
}

std::vector<std::vector<float>> res;
res.push_back(radius);
res.push_back(perimeter);
if(do_meandist) {
res.push_back(meandist);
}
return res;
}


