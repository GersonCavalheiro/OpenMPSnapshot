#pragma once

#include <cstddef>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "geodesic_mesh_elements.h"
#include "geodesic_constants.h"

namespace geodesic {

struct edge_visible_from_source
{
unsigned source;
edge_pointer edge;
};

class Mesh
{
public:
Mesh() {}

~Mesh() {}

template<class Points, class Faces>
void initialize_mesh_data(
const Points& p,
const Faces& tri,
const bool silent); 

std::vector<Vertex>& vertices() { return m_vertices; }
std::vector<Edge>& edges() { return m_edges; }
std::vector<Face>& faces() { return m_faces; }

unsigned closest_vertices(
const SurfacePoint* p,
std::vector<vertex_pointer>* storage = nullptr); 

private:
void build_adjacencies(const bool silent); 
bool verify(const bool silent);            

std::vector<Vertex> m_vertices;
std::vector<Edge> m_edges;
std::vector<Face> m_faces;
};

inline unsigned
Mesh::closest_vertices(const SurfacePoint* p, std::vector<vertex_pointer>* storage)
{
assert(p->type() != UNDEFINED_POINT);

if (p->type() == VERTEX) {
if (storage) {
storage->push_back(static_cast<vertex_pointer>(p->base_element()));
}
return 1;
} else if (p->type() == FACE) {
if (storage) {
const vertex_pointer vp = p->base_element()->adjacent_vertices().front();
storage->push_back(vp);
storage->push_back((vp + 1));
storage->push_back((vp + 2));
}
return 2;
} else if (p->type() == EDGE) 
{
const edge_pointer edge = static_cast<const edge_pointer>(p->base_element());

if (storage) {
storage->push_back(edge->adjacent_vertices()[0]);
storage->push_back(edge->adjacent_vertices()[1]);

for (unsigned i = 0; i < edge->adjacent_faces().size(); ++i) {
const face_pointer face = edge->adjacent_faces()[i];
storage->push_back(face->opposite_vertex(edge));
}
}
return 2 + edge->adjacent_faces().size();
}

assert(0);
return 0;
}

template<class Points, class Faces>
void
Mesh::initialize_mesh_data(
const Points& p,
const Faces& tri,
const bool silent) 
{
assert(p.size() % 3 == 0);
unsigned const num_vertices = p.size() / 3;
assert(tri.size() % 3 == 0);
unsigned const num_faces = tri.size() / 3;

m_vertices.resize(num_vertices);
for (unsigned i = 0; i < num_vertices; ++i) 
{
Vertex& v = m_vertices[i];
v.set_id(i);

unsigned shift = 3 * i;
v.set(p[shift], p[shift + 1], p[shift + 2]);
}

m_faces.resize(num_faces);
for (unsigned i = 0; i < num_faces; ++i) 
{
Face& f = m_faces[i];
f.set_id(i);
f.adjacent_vertices().resize(3); 

unsigned shift = 3 * i;
for (unsigned j = 0; j < 3; ++j) {
unsigned vertex_index = tri[shift + j];
assert(vertex_index < num_vertices);
f.adjacent_vertices()[j] = &m_vertices[vertex_index];
}
}

build_adjacencies(silent); 
}

inline void
Mesh::build_adjacencies(bool silent=true)
{
std::vector<unsigned> count(m_vertices.size()); 
for (unsigned i = 0; i < m_faces.size(); ++i) {
Face& f = m_faces[i];
for (unsigned j = 0; j < 3; ++j) {
unsigned vertex_id = f.adjacent_vertices()[j]->id();
assert(vertex_id < m_vertices.size());
count[vertex_id]++;
}
}

for (unsigned i = 0; i < m_vertices.size(); ++i) 
{
Vertex& v = m_vertices[i];
unsigned num_adjacent_faces = count[i];

v.adjacent_faces().resize(num_adjacent_faces);
}

std::fill(count.begin(), count.end(), 0);
for (unsigned i = 0; i < m_faces.size(); ++i) {
Face& f = m_faces[i];
for (unsigned j = 0; j < 3; ++j) {
vertex_pointer v = f.adjacent_vertices()[j];
v->adjacent_faces()[count[v->id()]++] = &f;
}
}

std::vector<HalfEdge> half_edges(m_faces.size() * 3);
unsigned k = 0;
for (unsigned i = 0; i < m_faces.size(); ++i) {
Face& f = m_faces[i];
for (unsigned j = 0; j < 3; ++j) {
half_edges[k].face_id = i;
unsigned vertex_id_1 = f.adjacent_vertices()[j]->id();
unsigned vertex_id_2 = f.adjacent_vertices()[(j + 1) % 3]->id();
half_edges[k].vertex_0 = std::min(vertex_id_1, vertex_id_2);
half_edges[k].vertex_1 = std::max(vertex_id_1, vertex_id_2);

k++;
}
}
std::sort(half_edges.begin(), half_edges.end());

unsigned number_of_edges = 1;
for (unsigned i = 1; i < half_edges.size(); ++i) {
if (half_edges[i] != half_edges[i - 1]) {
++number_of_edges;
} else {
if (i <
half_edges.size() - 1) 
{                          
assert(half_edges[i] != half_edges[i + 1]);
}
}
}

m_edges.resize(number_of_edges);
unsigned edge_id = 0;
for (unsigned i = 0; i < half_edges.size();) {
Edge& e = m_edges[edge_id];
e.set_id(edge_id++);

e.adjacent_vertices().resize(2); 

e.adjacent_vertices()[0] = &m_vertices[half_edges[i].vertex_0];
e.adjacent_vertices()[1] = &m_vertices[half_edges[i].vertex_1];

e.length() = e.adjacent_vertices()[0]->distance(*e.adjacent_vertices()[1]);
assert(e.length() > 1e-100); 

if (i != half_edges.size() - 1 && half_edges[i] == half_edges[i + 1]) 
{
e.adjacent_faces().resize(2);
e.adjacent_faces()[0] = &m_faces[half_edges[i].face_id];
e.adjacent_faces()[1] = &m_faces[half_edges[i + 1].face_id];
i += 2;
} else 
{
e.adjacent_faces().resize(1); 
e.adjacent_faces()[0] = &m_faces[half_edges[i].face_id];
i += 1;
}
}

std::fill(count.begin(), count.end(), 0);
for (unsigned i = 0; i < m_edges.size(); ++i) {
Edge& e = m_edges[i];
assert(e.adjacent_vertices().size() == 2);
count[e.adjacent_vertices()[0]->id()]++;
count[e.adjacent_vertices()[1]->id()]++;
}
for (unsigned i = 0; i < m_vertices.size(); ++i) {
m_vertices[i].adjacent_edges().resize(count[i]);
}
std::fill(count.begin(), count.end(), 0);
for (unsigned i = 0; i < m_edges.size(); ++i) {
Edge& e = m_edges[i];
for (unsigned j = 0; j < 2; ++j) {
vertex_pointer v = e.adjacent_vertices()[j];
v->adjacent_edges()[count[v->id()]++] = &e;
}
}

for (unsigned i = 0; i < m_faces.size(); ++i) {
m_faces[i].adjacent_edges().resize(3);
}

count.resize(m_faces.size());
std::fill(count.begin(), count.end(), 0);
for (unsigned i = 0; i < m_edges.size(); ++i) {
Edge& e = m_edges[i];
for (unsigned j = 0; j < e.adjacent_faces().size(); ++j) {
face_pointer f = e.adjacent_faces()[j];
assert(count[f->id()] < 3);
f->adjacent_edges()[count[f->id()]++] = &e;
}
}

auto cos_from_edges = [](double const a, double const b, double const c) -> double {
assert(a > 1e-50);
assert(b > 1e-50);
assert(c > 1e-50);

return std::min(std::max((b * b + c * c - a * a) / (2.0 * b * c), -1.0), 1.0);
};

for (unsigned i = 0; i < m_faces.size(); ++i) {
Face& f = m_faces[i];
double abc[3];
double sum = 0;
for (unsigned j = 0; j < 3; ++j) 
{
for (unsigned k = 0; k < 3; ++k) {
vertex_pointer v = f.adjacent_vertices()[(j + k) % 3];
abc[k] = f.opposite_edge(v)->length();
}

double angle = std::acos(cos_from_edges(abc[0], abc[1], abc[2]));
assert(angle > 1e-5); 

f.corner_angles()[j] = angle;
sum += angle;
}
assert(std::abs(sum - M_PI) < 1e-5); 
}

std::vector<double> total_vertex_angle(m_vertices.size());
for (unsigned i = 0; i < m_faces.size(); ++i) {
Face& f = m_faces[i];
for (unsigned j = 0; j < 3; ++j) {
vertex_pointer v = f.adjacent_vertices()[j];
total_vertex_angle[v->id()] += f.corner_angles()[j];
}
}

for (unsigned i = 0; i < m_vertices.size(); ++i) {
Vertex& v = m_vertices[i];
v.saddle_or_boundary() = (total_vertex_angle[v.id()] > 2.0 * M_PI - 1e-5);
}

for (unsigned i = 0; i < m_edges.size(); ++i) {
Edge& e = m_edges[i];
if (e.is_boundary()) {
e.adjacent_vertices()[0]->saddle_or_boundary() = true;
e.adjacent_vertices()[1]->saddle_or_boundary() = true;
}
}

assert(verify(silent));
}

inline bool
Mesh::verify(bool silent=true) 
{    
std::vector<bool> map(m_vertices.size(), false);
for (unsigned i = 0; i < m_edges.size(); ++i) {
edge_pointer e = &m_edges[i];
map[e->adjacent_vertices()[0]->id()] = true;
map[e->adjacent_vertices()[1]->id()] = true;
}
assert(std::find(map.begin(), map.end(), false) == map.end());

std::vector<face_pointer> stack(1, &m_faces[0]);
stack.reserve(m_faces.size());

map.resize(m_faces.size());
std::fill(map.begin(), map.end(), false);
map[0] = true;

while (!stack.empty()) {
face_pointer f = stack.back();
stack.pop_back();

for (unsigned i = 0; i < 3; ++i) {
edge_pointer e = f->adjacent_edges()[i];
face_pointer f_adjacent = e->opposite_face(f);
if (f_adjacent && !map[f_adjacent->id()]) {
map[f_adjacent->id()] = true;
stack.push_back(f_adjacent);
}
}
}
assert(std::find(map.begin(), map.end(), false) == map.end());

if(! silent) {
std::cout << "mesh has " << m_vertices.size() << " vertices, " << m_faces.size() << " faces, " << m_edges.size() << " edges\n";
}

unsigned total_boundary_edges = 0;
double longest_edge = 0;
double shortest_edge = 1e100;
for (unsigned i = 0; i < m_edges.size(); ++i) {
Edge& e = m_edges[i];
total_boundary_edges += e.is_boundary() ? 1 : 0;
longest_edge = std::max(longest_edge, e.length());
shortest_edge = std::min(shortest_edge, e.length());
}

if(! silent) {
std::cout << total_boundary_edges << " edges are boundary edges\n";
std::cout << "shortest/longest edges are " << shortest_edge << "/" << longest_edge << " = "
<< shortest_edge / longest_edge << std::endl;
}

double minx = 1e100;
double maxx = -1e100;
double miny = 1e100;
double maxy = -1e100;
double minz = 1e100;
double maxz = -1e100;
for (unsigned i = 0; i < m_vertices.size(); ++i) {
Vertex& v = m_vertices[i];
minx = std::min(minx, v.x());
maxx = std::max(maxx, v.x());
miny = std::min(miny, v.y());
maxy = std::max(maxy, v.y());
minz = std::min(minz, v.z());
maxz = std::max(maxz, v.z());
}

if(! silent) {
std::cout << "enclosing XYZ box:"
<< " X[" << minx << "," << maxx << "]"
<< " Y[" << miny << "," << maxy << "]"
<< " Z[" << minz << "," << maxz << "]" << std::endl;
}

double dx = maxx - minx;
double dy = maxy - miny;
double dz = maxz - minz;
if(! silent) {
std::cout << "approximate diameter of the mesh is " << std::sqrt(dx * dx + dy * dy + dz * dz)
<< std::endl;
}

double min_angle = 1e100;
double max_angle = -1e100;
for (unsigned i = 0; i < m_faces.size(); ++i) {
Face& f = m_faces[i];
for (unsigned j = 0; j < 3; ++j) {
double angle = f.corner_angles()[j];
min_angle = std::min(min_angle, angle);
max_angle = std::max(max_angle, angle);
}
}
if(! silent) {
std::cout << "min/max face angles are " << min_angle / M_PI * 180.0 << "/"
<< max_angle / M_PI * 180.0 << " degrees\n";

std::cout << std::endl;
}
return true;
}

} 
