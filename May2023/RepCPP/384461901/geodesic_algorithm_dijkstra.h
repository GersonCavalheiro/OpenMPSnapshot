#pragma once

#include "geodesic_algorithm_graph_base.h"
#include "geodesic_mesh_elements.h"
#include <vector>
#include <set>
#include <assert.h>

namespace geodesic {

class DijkstraNode
{
typedef DijkstraNode* node_pointer;

public:
DijkstraNode() {}
~DijkstraNode() {}

double& distance_from_source() { return m_distance; }
node_pointer& previous() { return m_previous; }
unsigned& source_index() { return m_source_index; }
vertex_pointer& vertex() { return m_vertex; }

void clear()
{
m_distance = GEODESIC_INF;
m_previous = nullptr;
}

bool operator()(node_pointer const s1, node_pointer const s2) const
{
return s1->distance_from_source() != s2->distance_from_source()
? s1->distance_from_source() < s2->distance_from_source()
: s1->vertex()->id() < s2->vertex()->id();
}

double distance(const SurfacePoint& p) const { return m_vertex->distance(p); }

SurfacePoint surface_point() { return SurfacePoint(m_vertex); }

private:
double m_distance;       
unsigned m_source_index; 
node_pointer m_previous; 
vertex_pointer m_vertex; 
};

class GeodesicAlgorithmDijkstra : public GeodesicAlgorithmGraphBase<DijkstraNode>
{
public:
typedef DijkstraNode Node;
typedef Node* node_pointer;

GeodesicAlgorithmDijkstra(geodesic::Mesh* mesh)
: GeodesicAlgorithmGraphBase<Node>(mesh)
{
m_nodes.resize(mesh->vertices().size());
for (unsigned i = 0; i < m_nodes.size(); ++i) {
m_nodes[i].vertex() = &m_mesh->vertices()[i];
}
}

~GeodesicAlgorithmDijkstra() {}

std::string name() const override { return "dijkstra"; }

protected:
void list_nodes_visible_from_source(MeshElementBase* p,
std::vector<node_pointer>& storage)
override; 

void list_nodes_visible_from_node(
node_pointer node, 
std::vector<node_pointer>& storage,
std::vector<double>& distances,
double threshold_distance) override; 
};

void
GeodesicAlgorithmDijkstra::list_nodes_visible_from_source(MeshElementBase* p,
std::vector<node_pointer>& storage)
{
assert(p->type() != UNDEFINED_POINT);

if (p->type() == FACE) {
face_pointer f = static_cast<face_pointer>(p);
for (unsigned i = 0; i < 3; ++i) {
vertex_pointer v = f->adjacent_vertices()[i];
storage.push_back(&m_nodes[node_index(v)]);
}
} else if (p->type() == EDGE) {
edge_pointer e = static_cast<edge_pointer>(p);
for (unsigned i = 0; i < 2; ++i) {
vertex_pointer v = e->adjacent_vertices()[i];
storage.push_back(&m_nodes[node_index(v)]);
}
} else 
{
vertex_pointer v = static_cast<vertex_pointer>(p);
storage.push_back(&m_nodes[node_index(v)]);
}
}

inline void
GeodesicAlgorithmDijkstra::list_nodes_visible_from_node(
node_pointer node, 
std::vector<node_pointer>& storage,
std::vector<double>& distances,
double threshold_distance)
{
vertex_pointer v = node->vertex();
assert(storage.size() == distances.size());

for (unsigned i = 0; i < v->adjacent_edges().size(); ++i) {
edge_pointer e = v->adjacent_edges()[i];
vertex_pointer new_v = e->opposite_vertex(v);
node_pointer new_node = &m_nodes[node_index(new_v)];

if (new_node->distance_from_source() > threshold_distance + e->length()) {
storage.push_back(new_node);
distances.push_back(e->length());
}
}
}

} 
