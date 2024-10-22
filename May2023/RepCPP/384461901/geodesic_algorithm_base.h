
#pragma once

#include "geodesic_mesh.h"
#include "geodesic_constants.h"
#include <iostream>
#include <ctime>

namespace geodesic {

class GeodesicAlgorithmBase
{
public:
GeodesicAlgorithmBase(geodesic::Mesh* mesh)
: m_max_propagation_distance(1e100)
, m_mesh(mesh)
{}

virtual ~GeodesicAlgorithmBase() {}

virtual void propagate(
const std::vector<SurfacePoint>& sources,
double max_propagation_distance = GEODESIC_INF, 
std::vector<SurfacePoint>* stop_points =
nullptr) = 0; 

virtual void trace_back(const SurfacePoint& destination, 
std::vector<SurfacePoint>& path) = 0;

virtual unsigned best_source(
const SurfacePoint& point, 
double& best_source_distance) = 0;

virtual void print_statistics() const 
{
std::cout << "propagation step took " << m_time_consumed << " seconds " << std::endl;
}

virtual std::string name() const = 0;

geodesic::Mesh* mesh() { return m_mesh; }

protected:
void set_stop_conditions(const std::vector<SurfacePoint>* stop_points, double stop_distance);
double stop_distance() { return m_max_propagation_distance; }

typedef std::pair<vertex_pointer, double> stop_vertex_with_distace_type;
std::vector<stop_vertex_with_distace_type>
m_stop_vertices; 
double m_max_propagation_distance; 

geodesic::Mesh* m_mesh;

double m_time_consumed; 
double
m_propagation_distance_stopped; 
};

inline void
GeodesicAlgorithmBase::set_stop_conditions(const std::vector<SurfacePoint>* stop_points,
double stop_distance)
{
m_max_propagation_distance = stop_distance;

if (!stop_points) {
m_stop_vertices.clear();
return;
}

m_stop_vertices.resize(stop_points->size());

std::vector<vertex_pointer> possible_vertices;
for (unsigned i = 0; i < stop_points->size(); ++i) {
const SurfacePoint* point = &(*stop_points)[i];

possible_vertices.clear();
m_mesh->closest_vertices(point, &possible_vertices);

vertex_pointer closest_vertex = nullptr;
double min_distance = 1e100;
for (unsigned j = 0; j < possible_vertices.size(); ++j) {
const double distance = point->distance(*possible_vertices[j]);
if (distance < min_distance) {
min_distance = distance;
closest_vertex = possible_vertices[j];
}
}
assert(closest_vertex);

m_stop_vertices[i].first = closest_vertex;
m_stop_vertices[i].second = min_distance;
}
}

} 
