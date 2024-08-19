

#pragma once

#include "Material.h"
#include "iso_triangle.hpp"
#include "mesh_tetra.hpp"
#include "mesh_vertex.hpp"

namespace bz_mesh {

enum class BandType { valence, conduction };

class MeshBZ {
private:

EmpiricalPseudopotential::Material m_material;


std::vector<Vertex> m_list_vertices;


std::vector<Tetra> m_list_tetrahedra;


std::vector<int> m_indices_valence_bands{};


std::vector<int> m_indices_conduction_bands{};


std::vector<double> m_min_band{};


std::vector<double> m_max_band{};

double m_total_volume = 0.0;

public:
MeshBZ(const EmpiricalPseudopotential::Material& material) : m_material(material){};

void read_mesh_geometry_from_msh_file(const std::string& filename);
void read_mesh_bands_from_msh_file(const std::string& filename);
void add_new_band_energies_to_vertices(const std::vector<double>& energies_at_vertices);
void compute_min_max_energies_at_tetras();
void compute_energy_gradient_at_tetras();

std::size_t get_number_vertices() const { return m_list_vertices.size(); }
std::size_t get_number_elements() const { return m_list_tetrahedra.size(); }
std::size_t get_number_bands() const { return m_min_band.size(); }

std::pair<double, double> get_min_max_energy_at_band(const int& band_index) const {
return std::make_pair(m_min_band[band_index], m_max_band[band_index]);
}

const std::vector<Vertex>& get_list_vertices() const { return m_list_vertices; }
const std::vector<Tetra>&  get_list_elements() const { return m_list_tetrahedra; }

double compute_mesh_volume() const;
double compute_iso_surface(double iso_energy, int band_index) const;
double compute_dos_at_energy_and_band(double iso_energy, int band_index) const;
double compute_overlapp_integral_impact_ionization_electrons(double energy);

std::vector<std::vector<double>> compute_dos_band_at_band(int         band_index,
double      min_energy,
double      max_energy,
int         num_threads,
std::size_t nb_points) const;
std::vector<std::vector<double>> compute_dos_band_at_band_auto(int band_index, std::size_t nb_points, int num_threads) const;

void export_k_points_to_file(const std::string& filename) const;
};

}  