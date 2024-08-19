
#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "mesh_vertex.hpp"

namespace bz_mesh {

class Tetra {
private:

std::size_t m_index;


std::array<Vertex*, 4> m_list_vertices{nullptr, nullptr, nullptr, nullptr};


std::array<vector3, 6> m_list_edges{};


double m_signed_volume;


std::size_t m_nb_bands = 0;


std::vector<double> m_min_energy_per_band;


std::vector<double> m_max_energy_per_band;

std::vector<double> m_gradient_energy_per_band;

public:
static std::vector<double> ms_case_stats;


Tetra() = delete;

Tetra(std::size_t index, const std::array<Vertex*, 4>& list_vertices);
void compute_min_max_energies_at_bands();

std::array<double, 4> get_band_energies_at_vertices(std::size_t index_band) const;

double  compute_signed_volume() const;
double  get_signed_volume() const { return m_signed_volume; }
vector3 compute_edge(std::size_t index_vtx_1, std::size_t index_vtx_2) const;
void    compute_gradient_energy_at_bands();

bool                  is_location_inside(const vector3& location) const;
std::array<double, 4> compute_barycentric_coordinates(const vector3& location) const;
vector3               compute_euclidean_coordinates(const std::array<double, 4>& barycentric_coordinates) const;
vector3               compute_euclidean_coordinates_with_indices(const std::array<double, 4>& barycentric_coordinates,
const std::array<int, 4>&    indices_vertex) const;

std::array<int, 4>   get_index_vertices_with_sorted_energy_at_band(std::size_t index_band) const;
std::vector<vector3> compute_band_iso_energy_surface(double iso_energy, std::size_t band_index) const;
double               compute_tetra_iso_surface_energy_band(double energy, std::size_t band_index) const;
double               compute_tetra_dos_energy_band(double energy, std::size_t band_index) const;

static void reset_stat_iso_computing() { ms_case_stats = std::vector<double>(5, 0.0); }

static void print_stat_iso_computing() {
std::size_t total_computation = ms_case_stats[0] + ms_case_stats[1] + ms_case_stats[2] + ms_case_stats[3] + ms_case_stats[4];
std::cout << "Case 1:       " << ms_case_stats[0] / double(total_computation) << std::endl;
std::cout << "Case 2:       " << ms_case_stats[1] / double(total_computation) << std::endl;
std::cout << "Case 3:       " << ms_case_stats[2] / double(total_computation) << std::endl;
std::cout << "Case 4:       " << ms_case_stats[3] / double(total_computation) << std::endl;
std::cout << "Case 5:       " << ms_case_stats[4] / double(total_computation) << std::endl;
std::cout << "Case Unknown: " << 1 - std::accumulate(ms_case_stats.begin(), ms_case_stats.end(), 0.0) / double(total_computation)
<< std::endl;
}
};

}  