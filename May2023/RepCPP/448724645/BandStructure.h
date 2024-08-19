#pragma once

#include <atomic>
#include <vector>

#include "Material.h"
#include "SymmetryPoints.h"
#include "Vector3D.h"

namespace EmpiricalPseudopotential {

class BandStructure {
public:
BandStructure() : m_nb_points(0), m_nb_bands(10), m_nearestNeighborsNumber(10), m_computation_time_s(-1.0){};


void Initialize(const Material&                 material,
std::size_t                     nb_bands,
const std::vector<std::string>& path,
unsigned int                    nrPoints,
unsigned int                    nearestNeighborsNumber,
bool                            enable_non_local_correction);


void Initialize(const Material&               material,
std::size_t                   nb_bands,
std::vector<Vector3D<double>> list_k_points,
unsigned int                  nearestNeighborsNumber,
bool                          enable_non_local_correction);

const std::vector<std::string>& GetPath() const { return m_path; }
unsigned int                    GetPointsNumber() const { return static_cast<unsigned int>(m_kpoints.size()); }
void                            Compute();
void                            Compute_parallel(int nb_threads);
double                          AdjustValues();

void        print_results() const;
std::string path_band_filename() const;
void        export_k_points_to_file(std::string filename) const;
void        export_result_in_file(const std::string& filename) const;
void        export_result_in_file_with_kpoints(const std::string& filename) const;

unsigned int        get_number_of_bands() const { return m_nb_bands; }
std::vector<double> get_band(unsigned int band_index) const;
double              get_energy_at_k_band(unsigned int band_index, unsigned int index_k) const { return m_results[index_k][band_index]; }

std::vector<Vector3D<double>>    get_kpoints() const { return m_kpoints; }
std::vector<std::vector<double>> get_band_energies() const { return m_results; }

private:
Materials materials;

SymmetryPoints            symmetryPoints;
std::vector<unsigned int> symmetryPointsPositions;
std::vector<std::string>  m_path;
unsigned int              m_nb_points;

Material     m_material;
unsigned int m_nb_bands;
unsigned int m_nearestNeighborsNumber;
bool         m_enable_non_local_correction;

std::vector<Vector3D<int>>       basisVectors;
std::vector<Vector3D<double>>    m_kpoints;
std::vector<std::vector<double>> m_results;

double m_computation_time_s;

static bool FindBandGap(const std::vector<std::vector<double>>& results, double& maxValValence, double& minValConduction);
bool        GenerateBasisVectors(unsigned int nearestNeighborsNumber);
};

void export_vector_bands_result_in_file(const std::string& filename, std::vector<std::vector<double>>);

}  
